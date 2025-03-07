PINECONE_API_KEY = 'pcsk_3otc4w_DdvWoFtzsyYgabEMqWnrnATWHK1dh3fBUR3YXABE47wUqHXb5CJnDaVJPuV2k2M'
DEEPSEEK_API_KEY = 'sk-ee79849dfbe84d588fda7f4e958d199c'
STREAMLIT_INDEX_NAME = 'rag-demo'
GEMINI_API_KEY = 'AIzaSyDBNILezgxR3qMjpQ5DGtapLhZJu_psZ58'

# creating the vector database
import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
import requests
import subprocess
from PyPDF2 import PdfReader
import pprint
import time
# from openai import OpenAI
import pickle

# Pinecone obj
pc = Pinecone(api_key=PINECONE_API_KEY)
# transformer
model = SentenceTransformer('all-mpnet-base-v2')
# path of resource pdf
pdf_path = 'experian_annual_report_2024_web.pdf'

# Vector db creation functions
# Define the index name and the dimension of our embedding model (768).
if not pc.has_index("rag-demo"):
    print("Creating index in pinecone...")
    pc.create_index(
        name="rag-demo",
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        ) 
    )

# Connect to the index
while not pc.describe_index("rag-demo").status['ready']:
    time.sleep(1)

def load_pdf(filepath):
    """
    Load a PDF file and return a list where each element is the text of a page.
    """
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        pages = [page.extract_text() for page in reader.pages]
    return pages



def generate_embedding(text):
    """
    Generate a 768-dimension embedding for the given text.
    """
    return model.encode(text).tolist()

pdf_pages = load_pdf(pdf_path)
print(f"Loaded {len(pdf_pages)} pages from the PDF.")
embeddings = [generate_embedding(page) for page in pdf_pages]
print("Generated embeddings for all pages.")

# Generate embeddings for each page of the PDF.
# if os.path.exists("embeddings.pkl"):
#     with open("embeddings.pkl", "rb") as f:
#         pdf_pages, embeddings = pickle.load(f)
#     print("Embeddings loaded from file.")
# else:
#     # Load the PDF and generate embeddings
#     pdf_pages = load_pdf(pdf_path)
#     print(f"Loaded {len(pdf_pages)} pages from the PDF.")
#     embeddings = [generate_embedding(page) for page in pdf_pages]
#     # Save embeddings to a file for future use
#     with open("embeddings.pkl", "wb") as f:
#         pickle.dump((pdf_pages, embeddings), f)
#     print("Embeddings generated and saved.")

# Prepare the vectors for upserting. Each vector is assigned a unique ID and metadata.
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
print(f"Upserted {len(vectors)} vectors into the Pinecone index.")

def query_pinecone(query_text, top_k=5):
    """
    Query the Pinecone index for the top_k most similar text chunks.
    """
    query_embedding = generate_embedding(query_text)
    print(query_embedding)
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return result

# LLM functions
test_query = "what is in your context given by the vector db?"
results = query_pinecone(test_query)
print("results: ", results)
context = ""
print("Query Results:")
for match in results['matches']:
    print(f"Score: {match['score']}\nText: {match['metadata']['text']}\n")
    context += f"Score: {match['score']}\nText: {match['metadata']['text']}\n"

print("Context: ", context)
from google import genai
def generate_response(prompt, context):
    """
    Call LLM to generate a response, using the context from Pinecone.
    """

    # client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    # response = client.chat.completions.create(
    #             model="deepseek-chat",
    #                 messages=[
    #                     {"role": "system", "content": "You are a expert of the given context, answer questions based on the given context. If you are not sure, say I dont know: " + context},
    #                     {"role": "user", "content": prompt},
    #             ],
    #             stream=False
    #         )
    # return response.choices[0].message.content
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=f"Context: {context}\\n\\nPrompt: {prompt}"
    )
    return response.text

pprint.pprint(generate_response(test_query, context))