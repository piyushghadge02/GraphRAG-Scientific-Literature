#!/bin/bash

# Phase 5 - Launch Script for GraphRAG Scientific Literature Demo

set -e

echo "========================================"
echo "GraphRAG Scientific Literature Demo"
echo "========================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

# Check if Neo4j is running
echo "🔍 Checking Neo4j connection..."
if ! python -c "
from neo4j import GraphDatabase
from src.config import settings
try:
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    with driver.session() as session:
        session.run('RETURN 1')
    driver.close()
    print('✅ Neo4j is running')
    exit(0)
except Exception as e:
    print(f'❌ Neo4j connection failed: {e}')
    print('Please start Neo4j: docker compose up -d')
    exit(1)
" 2>&1; then
    exit 1
fi

# Check if data is loaded
echo "🔍 Checking if graph data is loaded..."
CHUNK_COUNT=$(python -c "
from neo4j import GraphDatabase
from src.config import settings
driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
with driver.session() as session:
    result = session.run('MATCH (c:Chunk) RETURN count(c) as count')
    count = result.single()['count']
driver.close()
print(count)
" 2>&1)

if [ "$CHUNK_COUNT" -eq "0" ]; then
    echo "⚠️  No chunks found in database!"
    echo "Please run the pipeline first: python execute_pipeline.py"
    exit 1
else
    echo "✅ Found $CHUNK_COUNT chunks in database"
fi

echo ""
echo "========================================"
echo "Starting Streamlit Demo..."
echo "========================================"
echo ""
echo "🚀 Demo will open at: http://localhost:8501"
echo "📝 Example questions:"
echo "   - Simple: Do mitochondria play a role in programmed cell death?"
echo "   - Complex: What is the relationship between mitochondrial dysfunction and programmed cell death?"
echo ""
echo "Press Ctrl+C to stop the demo"
echo ""

# Launch Streamlit
streamlit run app/main.py
