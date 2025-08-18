import os
import time
import wikipediaapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# Configuration
# ----------------------------
RAW_DIR = "../data/wikipedia"   # local folder to cache pages
DB_FAISS_PATH = "../vectorstore"  # folder to save FAISS index
PAGES = [
    "Deep learning",
    "Neural network",
    "Convolutional neural network",
    "Recurrent neural network",
    "Transformer (machine learning model)",
    "Generative adversarial network"
]

# Ensure directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DB_FAISS_PATH, exist_ok=True)

# ----------------------------
# Step 1: Download & cache Wikipedia pages
# ----------------------------
print("[1/4] Downloading Wikipedia pages (with caching)...")
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='DeepLearningRAGBot/1.0 (your_email@example.com)'
)

documents = []
for title in PAGES:
    safe_filename = title.replace(" ", "_") + ".txt"
    file_path = os.path.join(RAW_DIR, safe_filename)

    # Load from cache if exists
    if os.path.exists(file_path):
        print(f"  ✓ Loaded cached: {title}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        # Retry 3 times if network fails
        for attempt in range(3):
            try:
                page = wiki_wiki.page(title)
                if page.exists():
                    content = page.text
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"  ✓ Downloaded: {title}")
                else:
                    content = ""
                    print(f"  ✗ Page not found: {title}")
                break
            except Exception as e:
                print(f"    ! Attempt {attempt+1} failed: {e}")
                time.sleep(5)
        else:
            content = ""
            print(f"    ! Failed to download after 3 attempts: {title}")

    if content:
        documents.append(content)

# ----------------------------
# Step 2: Split text into chunks
# ----------------------------
print("[2/4] Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = []
for doc in documents:
    chunks = text_splitter.split_text(doc)
    docs.extend(chunks)

# ----------------------------
# Step 3: Generate embeddings
# ----------------------------
print("[3/4] Generating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Step 4: Store in FAISS
# ----------------------------
print("[4/4] Saving to FAISS vector store...")
vectorstore = FAISS.from_texts(docs, embeddings)
vectorstore.save_local(DB_FAISS_PATH)

print("\n✅ Ingestion complete! Vector store saved at:", DB_FAISS_PATH)
