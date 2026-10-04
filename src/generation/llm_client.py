"""
Task 3 — LLM Client with Groq API Key Rotation
Provides LLMClient class wrapping Groq, OpenAI, and Ollama.
Includes automatic key rotation, rate-limit recovery, and failover
for the Groq backend.
"""

import time
from typing import Dict, List, Optional

from src.config import settings


class LLMClient:
    """Client for generating completions using Groq, OpenAI, or Ollama backends."""

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.backend = backend or settings.LLM_BACKEND
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature

        # Track the current active key index for Groq backend
        self._current_key_index = 0
        # Cache for instantiated groq clients to avoid rebuilds
        self._groq_clients: Dict[str, any] = {}

    def get_current_key_index(self) -> int:
        """Return the current active key index (for debugging and logging)."""
        return self._current_key_index

    def _get_groq_client(self, api_key: str):
        """Instantiate or reuse a Groq client for the given key."""
        if api_key not in self._groq_clients:
            from groq import Groq
            self._groq_clients[api_key] = Groq(api_key=api_key)
        return self._groq_clients[api_key]

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate completion for a given prompt (and optional system prompt).
        Applies rate limit (429) rotation only for the Groq backend.
        """
        backend_lower = self.backend.lower()

        if backend_lower == "groq":
            keys = settings.GROQ_API_KEYS
            if not keys:
                raise ValueError(
                    "No Groq API keys are configured. Please set at least GROQ_API_KEY1."
                )

            attempts = 0
            max_attempts = len(keys)

            while attempts < max_attempts:
                # Make sure the current index is within bounds (e.g. if keys changed)
                self._current_key_index = self._current_key_index % len(keys)
                current_key = keys[self._current_key_index]

                try:
                    client = self._get_groq_client(current_key)
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content

                except Exception as e:
                    # Check if error is a rate limit error (HTTP 429)
                    is_rate_limit = False

                    try:
                        import groq
                        if isinstance(e, groq.RateLimitError):
                            is_rate_limit = True
                    except ImportError:
                        pass

                    # Fallback attribute/string checking
                    status_code = getattr(e, "status_code", None)
                    if status_code == 429 or "429" in str(e) or "rate_limit" in str(e).lower():
                        is_rate_limit = True

                    if is_rate_limit:
                        print(
                            f"[llm_client] WARNING: Groq key at index {self._current_key_index} "
                            f"hit rate limit (429). Error: {e}"
                        )
                        # Advance key index and retry
                        self._current_key_index = (self._current_key_index + 1) % len(keys)
                        attempts += 1

                        if attempts >= max_attempts:
                            raise RuntimeError(
                                "All Groq API keys are rate-limited. Exhasted all available keys."
                            ) from e

                        print(f"[llm_client] Rotating to key index {self._current_key_index}. Retrying in 1.5s...")
                        time.sleep(1.5)
                    else:
                        # Re-raise other errors directly
                        raise e

            raise RuntimeError("All Groq API keys are rate-limited. Exhausted all available keys.")

        elif backend_lower == "openai":
            import openai
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        elif backend_lower == "ollama":
            import openai
            # Ollama compatibility client
            client = openai.OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")
