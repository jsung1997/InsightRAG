import os
import glob
import textwrap
import pandas as pd
from typing import List, Any, Dict

# LangChain / OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.agents import Tool, AgentType, initialize_agent

from scipy import stats

# ---------------------------------------------------------
# 1. LOAD ENV & GLOBALS
# ---------------------------------------------------------

load_dotenv()  # loads OPENAI_API_KEY if in .env

DATA_PATH = "data/survey.csv"
DOCS_PATH = "docs/*.txt"

# Load survey data once, globally (you can change this later)
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    print(f"[WARNING] {DATA_PATH} not found. Tools using df will fail.")
    df = pd.DataFrame()


# ---------------------------------------------------------
# 2. BUILD RAG INDEX (QUESTIONNAIRE + THEORY DOCS)
# ---------------------------------------------------------

def load_text_documents(pattern: str) -> List[Document]:
    docs = []
    for filepath in glob.glob(pattern):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(filepath)}))
    return docs


def build_vectorstore() -> FAISS:
    print("[INFO] Building vectorstore from docs/ ...")
    raw_docs = load_text_documents(DOCS_PATH)
    if not raw_docs:
        print("[WARNING] No documents found under docs/. RAG will return empty.")
        # create dummy doc so FAISS doesn't break
        raw_docs = [Document(page_content="No documents loaded.", metadata={"source": "empty"})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    split_docs = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings()  # uses text-embedding-3-large by default (depending on version)
    vs = FAISS.from_documents(split_docs, embeddings)
    return vs


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ---------------------------------------------------------
# 3. TOOL FUNCTIONS (STATS & RAG)
# ---------------------------------------------------------

def rag_search(query: str) -> str:
    """
    Simple RAG function: retrieve relevant chunks from docs and return as a text block.
    The agent will then use this as context when answering.
    """
    docs = retriever.get_relevant_documents(query)
    out_parts = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown")
        out_parts.append(f"[{i+1}] Source: {src}\n{d.page_content}\n")
    return "\n---\n".join(out_parts)


def describe_variables(columns: str) -> str:
    """
    Describe one or multiple columns from the survey DataFrame.
    columns: comma-separated column names, e.g. 'Q1,Q2,Q3'
    """
    if df.empty:
        return "No survey data loaded. Please ensure data/survey.csv exists."

    col_list = [c.strip() for c in columns.split(",") if c.strip()]
    missing = [c for c in col_list if c not in df.columns]
    if missing:
        return f"The following columns were not found in the dataset: {missing}\nAvailable columns: {list(df.columns)}"

    desc = df[col_list].describe().to_string()
    head = df[col_list].head(5).to_string()
    return textwrap.dedent(f"""
    Descriptive statistics for columns: {col_list}

    [Describe()]
    {desc}

    [Head(5)]
    {head}
    """)


def run_ttest(params: str) -> str:
    """
    Run an independent samples t-test on a numeric variable between two groups.
    params format (string): "value_col, group_col, group1, group2"
    Example: "acceptance_score, region, urban, rural"
    """
    if df.empty:
        return "No survey data loaded. Please ensure data/survey.csv exists."

    try:
        value_col, group_col, group1, group2 = [p.strip() for p in params.split(",")]
    except ValueError:
        return (
            "Invalid params format. Use: 'value_col, group_col, group1, group2'\n"
            "Example: 'acceptance_score, region, urban, rural'"
        )

    if value_col not in df.columns or group_col not in df.columns:
        return f"Columns not found. value_col: {value_col}, group_col: {group_col}\nAvailable: {list(df.columns)}"

    # Drop NA
    subset = df[[value_col, group_col]].dropna()

    g1 = subset[subset[group_col] == group1][value_col]
    g2 = subset[subset[group_col] == group2][value_col]

    if len(g1) < 3 or len(g2) < 3:
        return (
            f"Not enough data for t-test.\n"
            f"{group1} count: {len(g1)}, {group2} count: {len(g2)}"
        )

    t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)  # Welch's t-test
    mean_g1 = g1.mean()
    mean_g2 = g2.mean()

    return textwrap.dedent(f"""
    Independent samples t-test result:

    Value column   : {value_col}
    Group column   : {group_col}
    Group 1 (g1)   : {group1} (n={len(g1)}, mean={mean_g1:.3f})
    Group 2 (g2)   : {group2} (n={len(g2)}, mean={mean_g2:.3f})

    t-statistic    : {t_stat:.4f}
    p-value        : {p_value:.6f}

    Interpretation:
    - If p-value < 0.05, there is a statistically significant difference
      in {value_col} between {group1} and {group2}.
    """)


# ---------------------------------------------------------
# 4. SETUP LLM & AGENT WITH TOOLS
# ---------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",  # change to any model you have access to
    temperature=0.2,
)

tools = [
    Tool(
        name="RAG_Search_Docs",
        func=rag_search,
        description=(
            "Use this to retrieve relevant context from the questionnaire, codebook, or theory documents. "
            "Input should be a natural language query about definitions, constructs, or theoretical concepts."
        ),
    ),
    Tool(
        name="Describe_Survey_Variables",
        func=describe_variables,
        description=(
            "Use this to get descriptive statistics (count, mean, std, min, max, quartiles) "
            "for one or more survey variables (columns). "
            "Input should be a comma-separated list of column names, e.g. 'Q1,Q2,Q3' or 'acceptance_score'."
        ),
    ),
    Tool(
        name="Run_TTest_Between_Groups",
        func=run_ttest,
        description=(
            "Use this to test differences between two groups using an independent samples t-test. "
            "Input format: 'value_col, group_col, group1, group2'. "
            "Example: 'acceptance_score, region, urban, rural'."
        ),
    ),
]

# Agent with ReAct style
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# ---------------------------------------------------------
# 5. SYSTEM MESSAGE / AGENT ROLE
# ---------------------------------------------------------

SYSTEM_INSTRUCTIONS = """
You are InsightRAG, an autonomous survey analysis assistant.

Your goals:
1) Understand the user's research-style questions about a survey dataset.
2) When needed, use the tools:
   - RAG_Search_Docs: to consult the questionnaire and theory documents.
   - Describe_Survey_Variables: to inspect data distributions and variable summaries.
   - Run_TTest_Between_Groups: to compare means between two groups.
3) Combine numerical outputs from tools with theoretical context from RAG to produce
   clear, well-structured explanations.
4) Write in a professional but accessible academic tone. If the user asks in Korean,
   respond in Korean (e.g., '-다, -이다, -있다' style if appropriate). If they ask
   in English, respond in English.

Always:
- Explicitly mention what statistical tests you used.
- Report key numbers (means, n, p-values) when available.
- Provide both a brief summary and a more detailed interpretation.
"""

# We'll pass this as part of the first message each time.
# For simplicity, we prepend it to the user query.


def chat_with_agent():
    print("=== InsightRAG: Autonomous Survey Analyst ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Bye!")
            break

        full_prompt = SYSTEM_INSTRUCTIONS + "\n\nUser query:\n" + user_input
        try:
            response = agent.run(full_prompt)
        except Exception as e:
            response = f"[ERROR] {e}"

        print("\nInsightRAG:\n", response, "\n")


if __name__ == "__main__":
    chat_with_agent()
