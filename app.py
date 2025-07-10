import os
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader # For PDF parsing (install with pip install PyPDF2)
# from docx import Document # For DOCX parsing (install with pip install python-docx)

app = Flask(__name__)

# Global variables for FAISS index and document store
faiss_index = None
documents_store = [] # Stores {'text': 'chunk_text', 'source': 'filename.txt'}
model = None # Sentence Transformer model

# --- Configuration ---
DOCUMENTS_DIR = 'Data' # Directory where your documents are stored
TOP_N_CHUNKS = 3 # Number of top relevant chunks to retrieve

def initialize_model():
    """
    Initializes the Sentence Transformer model for generating embeddings.
    Note: This is an embedding model, not a generative LLM like TinyLlama.
    """
    global model
    if model is None:
        print("Loading Sentence Transformer model (Hugging Face 'all-MiniLM-L6-v2')...")
        # This model is from Hugging Face and is specifically designed for embeddings.
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")

def load_and_process_documents():
    """
    Loads documents from the DOCUMENTS_DIR, processes them into chunks,
    generates embeddings, and builds a FAISS index.
    """
    global faiss_index, documents_store

    documents_store = []
    all_embeddings = []

    print(f"Loading documents from: {DOCUMENTS_DIR}")
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"'{DOCUMENTS_DIR}' directory created. Please add your sample .txt, .pdf, or .docx files here.")


    for filename in os.listdir(DOCUMENTS_DIR):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        content = ""
        try:
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif filename.endswith('.pdf'):
                # Basic PDF parsing using PyPDF2
                reader = PdfReader(filepath)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            # elif filename.endswith('.docx'):
            #     # Basic DOCX parsing using python-docx (uncomment if you install python-docx)
            #     doc = Document(filepath)
            #     for para in doc.paragraphs:
            #         content += para.text + "\n"
            else:
                print(f"Skipping unsupported file type: {filename}")
                continue

            # Simple chunking: split by sentences. For a robust system, consider more advanced chunking.
            chunks = [chunk.strip() for chunk in content.split('.') if chunk.strip()]
            for chunk in chunks:
                documents_store.append({'text': chunk, 'source': filename})
                all_embeddings.append(model.encode(chunk))

            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_embeddings:
        print("No documents processed. FAISS index will not be built.")
        faiss_index = None
        return

    # Convert embeddings to a numpy array for FAISS
    all_embeddings = np.array(all_embeddings).astype('float32')

    # Build FAISS index for efficient similarity search
    dimension = all_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension) # Using L2 distance
    faiss_index.add(all_embeddings)

    print(f"FAISS index built with {faiss_index.ntotal} embeddings.")

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles user queries:
    1. Embeds the query.
    2. Retrieves top N relevant document chunks from FAISS.
    3. Simulates LLM response by concatenating chunks.
    4. Includes citations.
    """
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'response': 'Please provide a query.'}), 400

    if faiss_index is None or not documents_store:
        return jsonify({'response': 'Document index not ready. Please ensure documents are loaded in the "Data" folder.'}), 503

    # Generate embedding for the user query
    query_embedding = model.encode(user_query).astype('float32').reshape(1, -1)

    # Perform similarity search in FAISS
    D, I = faiss_index.search(query_embedding, TOP_N_CHUNKS) # D = distances, I = indices

    retrieved_chunks = []
    sources = set()
    for i in I[0]:
        if i != -1 and i < len(documents_store): # Ensure index is valid and within bounds
            chunk_info = documents_store[i]
            retrieved_chunks.append(chunk_info['text'])
            sources.add(chunk_info['source'])

    if not retrieved_chunks:
        response_text = "I couldn't find relevant information in the documents for your query. Please try rephrasing."
        citations = ""
    else:
        # Simulate LLM response: Concatenate retrieved chunks.
        # In a full RAG system with an LLM API, the LLM would synthesize
        # a more natural answer from these chunks.
        response_text = "Based on the documents, here's what I found:\n\n" + "\n\n".join(retrieved_chunks)
        citations = "\n\nSources: " + ", ".join(sorted(list(sources)))

    final_response = response_text + citations
    return jsonify({'response': final_response})

if __name__ == '__main__':
    initialize_model()
    load_and_process_documents()
    # Run the Flask app
    app.run(debug=True)

