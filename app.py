from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import streamlit as st
from google.colab import files

#Function to flatten metadata
def flatten_metadata(metadata):
    flattened = {}
    for k, v in metadata.items():
        if isinstance(v, (dict, list)):
            flattened[k] = json.dumps(v, ensure_ascii=False)
        else:
            flattened[k] = v
    return flattened

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="tiiuae/falcon-7b-instruct")

# Load CAF chunks

uploaded = files.upload()
try:
    with open("caf_chunks.json", "r") as f:
        chunks = json.load(f)
except FileNotFoundError:
    st.error("Processed data file not found. Please run create_chunks.py first.")
    chunks = []

texts = [c["content"] for c in chunks]
metadatas = [flatten_metadata(c["metadata"]) for c in chunks]

# Build Chroma vectorstore
db = Chroma.from_texts(
    texts,
    embeddings,
    metadatas=metadatas,
    collection_name="caf_collection",
    persist_directory="./chroma_db"
)
db.persist()

# LLM setup
model_name = "tiiuae/falcon-7b-instruct"  # or mistralai/Mistral-7B-Instruct-v0.2
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=False,
    max_new_tokens=200,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=generator)

# Base retrievers
bm25_retriever = BM25Retriever.from_texts(texts, metadatas)
dense_retriever = db.as_retriever(search_kwargs={"k": 8})

# Hybrid retrieval (dense + BM25)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

# Final RetrievalQA chain using hybrid retriever directly
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=hybrid_retriever
)

system_prompt = """
You are a cybersecurity assistant for Cyberland Park.
Answer clearly, concisely, and only once.
Do not repeat phrases.
Include relevant MITRE techniques with their names and IDs if possible.
"""

def build_prompt(user_question, context=""):
    return f"{system_prompt}\nContext:\n{context}\nUser question: {user_question}"

print("Cyberland Park Cybersecurity Chatbot (type 'exit' to quit)")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    # build the prompt (no undefined variables)
    prompt = build_prompt(query)

    # invoke the QA chain
    response = qa.invoke(prompt)

    # handle dict or string outputs safely
    if isinstance(response, dict) and "result" in response:
        answer = response["result"]
    else:
        answer = str(response)

    print("\n--- Cyberland Park Assistant ---")
    print(answer.strip())
    print("-------------------------------\n")
