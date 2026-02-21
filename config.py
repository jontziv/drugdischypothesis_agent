import os
from dotenv import load_dotenv

load_dotenv()

# GROQ
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.4
GROQ_MAX_TOKENS = 4096

# Vector store
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
LITERATURE_COLLECTION = "biomedical_literature"
MOLECULE_COLLECTION = "molecules"

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# PubMed
PUBMED_EMAIL = "drugdiscovery_agent@research.org"
PUBMED_MAX_RESULTS = 20

# ChEMBL
CHEMBL_MAX_RESULTS = 30

# Elo
ELO_DEFAULT_RATING = 1500
ELO_K_FACTOR = 32

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HYPOTHESES_FILE = os.path.join(DATA_DIR, "hypotheses.json")
ELO_FILE = os.path.join(DATA_DIR, "elo_ratings.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_log.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
