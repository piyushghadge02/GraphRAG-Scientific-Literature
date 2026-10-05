from Bio import Entrez

Entrez.email = 'test@example.com'

# Test with a known PubMedQA context
abstract = 'Programmed cell death is the regulated death of cells within an organism and the lace plant provides a model system for studying developmental cell death processes in detail across leaf formation stages.'
# Try searching with key terms
terms_to_try = [
    f'"{abstract[:80]}"[Abstract]',  # Exact phrase
    'lace plant programmed cell death leaf formation[Abstract]',
    'Aponogeton madagascariensis programmed cell death[Abstract]',
]

for term in terms_to_try:
    print(f'Search term: {term}')
    handle = Entrez.esearch(db='pubmed', term=term, retmax=5)
    result = Entrez.read(handle)
    print(f'  Count: {result["Count"]}')
    print(f'  IDs: {result["IdList"]}')