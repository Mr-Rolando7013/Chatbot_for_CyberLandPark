from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import streamlit as st

# 🔧 Helper function to flatten metadata
def flatten_metadata(metadata):
    flattened = {}
    for k, v in metadata.items():
        if isinstance(v, (dict, list)):
            flattened[k] = json.dumps(v, ensure_ascii=False)
        else:
            flattened[k] = v
    return flattened

# ✅ Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="tiiuae/falcon-7b-instruct")

# ✅ Load CAF chunks
try:
    with open("caf_chunks.json", "r") as f:
        chunks = json.load(f)
except FileNotFoundError:
    st.error("Processed data file not found. Please run create_chunks.py first.")
    chunks = []

texts = [c["content"] for c in chunks]
metadatas = [flatten_metadata(c["metadata"]) for c in chunks]

# ✅ Build Chroma vectorstore
db = Chroma.from_texts(
    texts,
    embeddings,
    metadatas=metadatas,
    collection_name="caf_collection",
    persist_directory="./chroma_db"
)
db.persist()

# ✅ LLM setup
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
    do_sample=False,       # use greedy decoding
    max_new_tokens=200,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=generator)

# ✅ Base retrievers
bm25_retriever = BM25Retriever.from_texts(texts, metadatas)
dense_retriever = db.as_retriever(search_kwargs={"k": 8})

# ✅ Hybrid retrieval (dense + BM25)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

# ✅ Final RetrievalQA chain using hybrid retriever directly
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=hybrid_retriever
)

# 🎨 Streamlit UI
st.title("Cyberland Park Cybersecurity Chatbot 🤖")
st.caption("Ask questions about CAF and MITRE ATT&CK Frameworks")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me about CAF, MITRE, or cybersecurity policies...")

if user_input:
    prompt = f"""
    You are a cybersecurity assistant helping Cyberland Park improve resilience using the CAF and MITRE ATT&CK frameworks.
    Use the provided context if available.
    Respond clearly, accurately, and without repetition.
    Include relevant MITRE techniques with IDs if applicable.

    User question: {user_input}
    """
    response = qa.invoke(prompt)
    st.session_state.history.append((user_input, response))

# Display chat history
for message, answer in st.session_state.history:
    with st.chat_message("user"):
        st.write(message)
    with st.chat_message("assistant"):
        st.write(answer)
