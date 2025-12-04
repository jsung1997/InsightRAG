import os
import glob
import textwrap
from typing import List

import pandas as pd
from scipy import stats

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GPT4All


# =========================================================
# 1. PATHS & CONFIG
# =========================================================

DATA_PATH = "data/survey.csv"
DOCS_GLOB = "docs/*.txt"

# You have TWO options for your local model path:
# 1) Set an environment variable INSIGHTRAG_MODEL_PATH
#    Example (PowerShell):
#      $env:INSIGHTRAG_MODEL_PATH = "C:\\Users\\YOU\\AppData\\Local\\nomic.ai\\GPT4All\\your-model.gguf"
# 2) Or edit DEFAULT_MODEL_PATH below to your actual absolute path.

DEFAULT_MODEL_PATH = r"C:\path\to\your\local\model.gguf"  # <- change or override via ENV


def get_model_path() -> str:
    env_path = os.getenv("INSIGHTRAG_MODEL_PATH")
    if env_path:
        return env_path
    return DEFAULT_MODEL_PATH


MODEL_PATH = get_model_path()


# =========================================================
# 2. LOAD SURVEY DATA
# =========================================================

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded survey data from {DATA_PATH} with shape {df.shape}")
else:
    print(f"[WARNING] {DATA_PATH} not found. Data tools will not work until you add a CSV.")
    df = pd.DataFrame()


# =========================================================
# 3. BUILD RAG INDEX FROM DOCS/
# =========================================================

def load_text_documents(pattern: str) -> List[Document]:
    docs = []
    for filepath in glob.glob(pattern):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(filepath)}))
    return docs


def build_vectorstore() -> FAISS:
    print("[INFO] Building vector store from docs/ ...")
    raw_docs = load_text_documents(DOCS_GLOB)

    if not raw_docs:
        print("[WARNING] No .txt documents found in docs/. Creating a dummy document.")
        raw_docs = [Document(page_content="No documents provided.", metadata={"source": "empty"})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    split_docs = splitter.split_documents(raw_docs)

    # Free local embedding model (downloaded once)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# =========================================================
# 4. TOOL FUNCTIONS (RAG + STATS)
# =========================================================

def rag_search(query: str) -> str:
    """
    Retrieve relevant chunks from questionnaire/theory docs.
    The agent uses this for definitions and conceptual context.
    """
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."

    out_parts = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown")
        out_parts.append(f"[{i+1}] Source: {src}\n{d.page_content}\n")
    return "\n---\n".join(out_parts)


def describe_variables(columns: str) -> str:
    """
    Show descriptive statistics for one or more columns.
    Input: 'col1,col2,col3'
    """
    if df.empty:
        return "No survey data loaded. Please add data/survey.csv."

    col_list = [c.strip() for c in columns.split(",") if c.strip()]
    missing = [c for c in col_list if c not in df.columns]

    if missing:
        return (
            f"The following columns were not found: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    desc = df[col_list].describe(include="all").to_string()
    head = df[col_list].head(5).to_string()

    return textwrap.dedent(f"""
    Descriptive statistics for columns: {col_list}

    [describe()]
    {desc}

    [head(5)]
    {head}
    """)


def run_ttest(params: str) -> str:
    """
    Independent samples t-test between two groups.

    Input format:
      "value_col, group_col, group1, group2"

    Example:
      "acceptance_score, region, urban, rural"
    """
    if df.empty:
        return "No survey data loaded. Please add data/survey.csv."

    try:
        value_col, group_col, group1, group2 = [p.strip() for p in params.split(",")]
    except ValueError:
        return (
            "Invalid input. Use: 'value_col, group_col, group1, group2'\n"
            "Example: 'acceptance_score, region, urban, rural'"
        )

    if value_col not in df.columns or group_col not in df.columns:
        return (
            f"Columns not found. value_col: {value_col}, group_col: {group_col}\n"
            f"Available columns: {list(df.columns)}"
        )

    subset = df[[value_col, group_col]].dropna()

    g1 = subset[subset[group_col] == group1][value_col]
    g2 = subset[subset[group_col] == group2][value_col]

    if len(g1) < 3 or len(g2) < 3:
        return (
            "Not enough data for a t-test.\n"
            f"{group1} count: {len(g1)}, {group2} count: {len(g2)}"
        )

    t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)  # Welch t-test
    mean_g1 = g1.mean()
    mean_g2 = g2.mean()

    return textwrap.dedent(f"""
    Independent samples t-test result:

    Value column : {value_col}
    Group column : {group_col}

    Group 1: {group1}
        n    = {len(g1)}
        mean = {mean_g1:.3f}

    Group 2: {group2}
        n    = {len(g2)}
        mean = {mean_g2:.3f}

    t-statistic = {t_stat:.4f}
    p-value     = {p_value:.6f}

    Interpretation guideline:
    - If p-value < 0.05, there is a statistically significant difference in
      {value_col} between {group1} and {group2}.
    """)


# =========================================================
# 5. LOCAL LLM (GPT4ALL) + AGENT SETUP
# =========================================================

if not os.path.exists(MODEL_PATH):
    print(f"[WARNING] Model file not found at:\n  {MODEL_PATH}")
    print("Set INSIGHTRAG_MODEL_PATH env var or edit DEFAULT_MODEL_PATH in main.py.")
    # We still try to load; GPT4All will raise a clearer error.

llm = GPT4All(
    model=MODEL_PATH,
    verbose=True,
    n_threads=8,   # adjust based on your CPU
)

tools = [
    Tool(
        name="RAG_Search_Docs",
        func=rag_search,
        description=(
            "Use this to look up information in the questionnaire, codebook, or theory documents. "
            "Input is a natural language query about definitions, constructs, or theoretical concepts."
        ),
    ),
    Tool(
        name="Describe_Survey_Variables",
        func=describe_variables,
        description=(
            "Use this to get descriptive statistics for one or more survey variables. "
            "Input must be a comma-separated list of column names, e.g. 'Q1,Q2,Q3'."
        ),
    ),
    Tool(
        name="Run_TTest_Between_Groups",
        func=run_ttest,
        description=(
            "Use this to compare the mean of a numeric variable between two groups "
            "with an independent samples t-test. "
            "Input format: 'value_col, group_col, group1, group2'."
        ),
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

SYSTEM_INSTRUCTIONS = """
You are InsightRAG, an autonomous survey analysis assistant.

Your goals:
1) Understand the user's questions about a survey dataset and related theory.
2) Use tools when needed:
   - RAG_Search_Docs: to consult questionnaire, codebook, and theory notes.
   - Describe_Survey_Variables: to check distributions and descriptive stats.
   - Run_TTest_Between_Groups: to test differences between two groups.
3) Combine tool outputs with clear reasoning to produce well-structured answers.

Always:
- Explain which statistical methods you used.
- Report key numbers (sample sizes, means, p-values) when available.
- Answer in clear, professional English.
"""


# =========================================================
# 6. SIMPLE COMMAND-LINE CHAT LOOP
# =========================================================

def chat_with_agent():
    print("=== InsightRAG: Autonomous Survey Analyst (Local LLM) ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        prompt = SYSTEM_INSTRUCTIONS + "\n\nUser query:\n" + user_input
        try:
            response = agent.run(prompt)
        except Exception as e:
            response = f"[ERROR] {e}"

        print("\nInsightRAG:\n", response, "\n")


if __name__ == "__main__":
    chat_with_agent()
