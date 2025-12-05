import os
import glob
import textwrap
import pandas as pd
from typing import List

# GPT4All Local LLM (updated import)
from langchain_community.llms import GPT4All

# LangChain tools & agent
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Local embeddings (offline)
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

# Stats
from scipy import stats


# ---------------------------------------------------------
# 1. FILE PATHS AND DATA LOADING
# ---------------------------------------------------------

DATA_PATH = "data/survey.csv"
DOCS_PATH = "docs/*.txt"

# Load dataset
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    print(f"[WARNING] Missing dataset at {DATA_PATH}")
    df = pd.DataFrame()


# ---------------------------------------------------------
# 2. BUILD LOCAL RAG INDEX (OFFLINE)
# ---------------------------------------------------------

def load_docs():
    docs = []
    for f in glob.glob(DOCS_PATH):
        with open(f, "r", encoding="utf-8") as doc:
            text = doc.read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(f)}))
    return docs


def build_vectorstore():
    print("[INFO] Building vectorstore...")

    raw_docs = load_docs()
    if not raw_docs:
        raw_docs = [Document(page_content="No documents available.", metadata={"source": "empty"})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    # Local embeddings model (offline)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts)

    return FAISS.from_embeddings(embeddings, chunks)


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ---------------------------------------------------------
# 3. TOOL FUNCTIONS (RAG + STATS)
# ---------------------------------------------------------

def rag_search(query: str) -> str:
    """Retrieve relevant content from Q&A documents."""
    docs = retriever.get_relevant_documents(query)
    result = []
    for d in docs:
        result.append(f"Source: {d.metadata['source']}\n{d.page_content}\n")
    return "\n---\n".join(result)


def describe_variables(cols: str) -> str:
    """Return descriptive stats for given survey variables."""
    if df.empty:
        return "Dataset not loaded. Add survey.csv"

    col_list = [c.strip() for c in cols.split(",")]
    missing = [c for c in col_list if c not in df.columns]

    if missing:
        return f"Missing columns: {missing}\nAvailable: {list(df.columns)}"

    desc = df[col_list].describe().to_string()
    head = df[col_list].head().to_string()

    return f"[Describe]\n{desc}\n\n[Head]\n{head}"


def run_ttest(params: str) -> str:
    """Run t-test between two groups: 'value_col, group_col, A, B'."""
    if df.empty:
        return "Dataset not loaded."

    try:
        value_col, group_col, g1, g2 = [x.strip() for x in params.split(",")]
    except:
        return "Format: value_col, group_col, group1, group2"

    if value_col not in df.columns or group_col not in df.columns:
        return f"Columns not found. Available: {list(df.columns)}"

    sub = df[[value_col, group_col]].dropna()
    v1 = sub[sub[group_col] == g1][value_col]
    v2 = sub[sub[group_col] == g2][value_col]

    if len(v1) < 3 or len(v2) < 3:
        return "Not enough data for t-test."

    t, p = stats.ttest_ind(v1, v2, equal_var=False)

    return textwrap.dedent(f"""
    T-test result:
    Group {g1}: n={len(v1)}, mean={v1.mean():.3f}
    Group {g2}: n={len(v2)}, mean={v2.mean():.3f}
    t = {t:.4f}, p = {p:.6f}

    Interpretation:
    p < 0.05 → significant difference.
    """)


# ---------------------------------------------------------
# 4. INITIALIZE LOCAL LLM (DEEPSEEK-R1)
# ---------------------------------------------------------

llm = GPT4All(
    model=r"C:\Users\Jay\AppData\Local\nomic.ai\GPT4All\DeepSeek-R1-Distill-Qwen-7B-Q4_0.gguf",
    max_tokens=4096,
    temperature=0.2,
    verbose=False
)


# ---------------------------------------------------------
# 5. AGENT SETUP WITH TOOLS
# ---------------------------------------------------------

tools = [
    Tool(
        name="RAG_Search",
        func=rag_search,
        description="Search questionnaire & theory documents for definitions, context, or constructs."
    ),
    Tool(
        name="Describe_Variables",
        func=describe_variables,
        description="Describe survey variables. Input: 'Q1,Q2,Q3'"
    ),
    Tool(
        name="Run_TTest",
        func=run_ttest,
        description="Run t-test. Format: 'value_col, group_col, group1, group2'"
    ),
]

SYSTEM_PROMPT = """
You are InsightRAG, an autonomous survey analysis agent.
- Use tools when needed.
- Combine numerical results with theory context.
- When user speaks Korean, respond in Korean academic style ('-다, -이다, -있다').
- Provide structured, accurate explanations.
"""


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)


# ---------------------------------------------------------
# 6. CLI CHAT LOOP
# ---------------------------------------------------------

def chat():
    print("=== InsightRAG (Local DeepSeek Model) ===")
    print("Type 'exit' to quit.\n")

    while True:
        user = input("You: ")
        if user.lower() == "exit":
            break

        prompt = SYSTEM_PROMPT + "\nUser:\n" + user
        try:
            answer = agent.run(prompt)
        except Exception as e:
            answer = f"ERROR: {e}"

        print("\nInsightRAG:\n", answer, "\n")


if __name__ == "__main__":
    chat()
