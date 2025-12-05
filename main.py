from langchain_community.chat_models import ChatGPT4All
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
import glob
import os

# ===== Load Local LLM =====
llm = ChatGPT4All(
    model=r"C:\Users\Jay\AppData\Local\nomic.ai\GPT4All\DeepSeek-R1-Distill-Qwen-7B-Q4_0.gguf",
    max_tokens=4096,
    temperature=0.2
)

# ===== Build Vectorstore from docs =====
def load_docs():
    docs = []
    for f in glob.glob("docs/*.txt"):
        text = open(f, "r", encoding="utf-8").read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(f)}))
    return docs

from sentence_transformers import SentenceTransformer
import faiss

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_vectorstore():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    embeddings = embed_model.encode(texts)
    index = FAISS.from_embeddings(embeddings, chunks)
    return index

vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever()

# ===== Define Tools =====
def rag_search(q):
    docs = retriever.get_relevant_documents(q)
    return "\n\n".join([d.page_content for d in docs])

def describe(col):
    df = pd.read_csv("data/survey.csv")
    return df[col.split(",")].describe().to_string()

tools = [
    Tool(name="RAG_Search", func=rag_search, description="Retrieve context from theory docs"),
    Tool(name="Describe", func=describe, description="Describe survey variables")
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# ===== Test Query =====
response = agent.run("Explain the meaning of normative pillar based on the documents.")
print("\nInsightRAG:", response)
