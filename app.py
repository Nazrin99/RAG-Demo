import streamlit as st
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import pickle
from google import genai
import os

# Your API keys (make sure to secure them in production!)
PINECONE_API_KEY="pcsk_3otc4w_DdvWoFtzsyYgabEMqWnrnATWHK1dh3fBUR3YXABE47wUqHXb5CJnDaVJPuV2k2M"
GEMINI_API_KEY="AIzaSyDBNILezgxR3qMjpQ5DGtapLhZJu_psZ58"
STREAMLIT_INDEX_NAME = 'rag-demo'

# Initialize Pinecone and SentenceTransformer
pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load or generate embeddings
pdf_path = 'experian_annual_report_2024_web.pdf'

@st.cache_data(show_spinner=False, persist=True)
def load_embeddings():
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            pdf_pages, embeddings = pickle.load(f)
        print("Embeddings loaded from file.")
    else:
        pdf_pages = load_pdf(pdf_path)
        print(f"Loaded {len(pdf_pages)} pages from the PDF.")
        embeddings = [generate_embedding(page) for page in pdf_pages]
        with open("embeddings.pkl", "wb") as f:
            pickle.dump((pdf_pages, embeddings), f)
        print("Embeddings generated and saved.")
    return pdf_pages, embeddings

def load_pdf(filepath):
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        pages = [page.extract_text() for page in reader.pages]
    return pages

def generate_embedding(text):
    return model.encode(text).tolist()

pdf_pages, embeddings = load_embeddings()

# Prepare Pinecone
if not pc.has_index(STREAMLIT_INDEX_NAME):
    pc.create_index(
        name=STREAMLIT_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

vectors = [
    {
        'id': f'page_{i}',
        'values': embedding,
        'metadata': {'page_number': i, 'text': pdf_pages[i]}
    }
    for i, embedding in enumerate(embeddings)
]

index = pc.Index("rag-demo")
index.upsert(vectors)

# Streamlit UI
st.title("VectorDB RAG Chatbot")
st.write("Ask me questions about Experian's performance!")

query = st.text_input("Enter your query:")
if query:
    query_embedding = generate_embedding(query)
    result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    context = "\n".join([match["metadata"]["text"] for match in result['matches']])

    # Generate response
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Context: {context}\n\nPrompt: {query}"
    )
    st.write(f"**Response:** {response.text}")
