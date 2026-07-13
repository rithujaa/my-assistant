import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import chromadb

# ── SETUP ─────────────────────────────────────────────────────────
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Ask About Rithujaa", page_icon="👩‍💻", layout="centered")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        padding: 25px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 1.8rem; margin: 0; }
    .main-header p { color: #a8b2d8; margin: 8px 0 0 0; font-size: 0.95rem; }
    .suggestion-btn { margin: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>👩‍💻 Ask About Rithujaa</h1>
    <p>Ask me anything about Rithujaa's experience, projects, skills, or background.</p>
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_base_built" not in st.session_state:
    st.session_state.knowledge_base_built = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ── DOCUMENT PATHS ────────────────────────────────────────────────
DOCUMENT_PATHS = [
    "01_professional_experience.txt",
    "02_projects.txt",
    "03_skills_and_tools.txt",
    "04_education.txt",
    "05_personal_background.txt",
    "06_goals_and_job_search.txt",
]

# ── BUILD KNOWLEDGE BASE ──────────────────────────────────────────
def build_knowledge_base():
    """Load all documents, chunk them, embed and store in ChromaDB"""
    all_text = ""
    for path in DOCUMENT_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(all_text)

    # Embed and store in ChromaDB
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model="text-embedding-ada-002"
    )
    chroma_client = chromadb.EphemeralClient()
    vectorstore = Chroma.from_texts(
        chunks,
        embeddings,
        client=chroma_client,
        collection_name="rithujaa_profile"
    )
    return vectorstore


def answer_question(question, vectorstore, chat_history):
    """Retrieve relevant chunks and generate answer using GPT-4o-mini"""

    # Build conversation history context
    history_text = ""
    if chat_history:
        history_text = "Previous conversation:\n"
        for item in chat_history[-3:]:
            history_text += f"Q: {item['question']}\nA: {item['answer']}\n\n"

    # Retrieve most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build prompt
    prompt = f"""You are a helpful assistant that answers questions about Rithujaa Rajendrakumar based on the information provided below.

Answer naturally and conversationally. Be specific and accurate. If the information isn't in the context, say you don't have that detail.

{history_text}

Context about Rithujaa:
{context}

Question: {question}

Answer:"""

    # Generate answer using GPT-4o-mini
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key
    )
    response = llm.invoke(prompt)
    return response.content.strip()


# ── BUILD KNOWLEDGE BASE ON STARTUP ───────────────────────────────
if not st.session_state.knowledge_base_built:
    with st.spinner("Loading Rithujaa's profile..."):
        st.session_state.vectorstore = build_knowledge_base()
        st.session_state.knowledge_base_built = True

# ── SUGGESTED QUESTIONS ───────────────────────────────────────────
st.markdown("### 💡 Try asking:")
col1, col2 = st.columns(2)
suggestions = [
    "What is Rithujaa's experience with Snowflake?",
    "Tell me about the AI BI Assistant project",
    "What ML models has she worked with?",
    "What is her educational background?",
    "What kind of roles is she looking for?",
    "What tools and languages does she know?",
]
for i, suggestion in enumerate(suggestions):
    with col1 if i % 2 == 0 else col2:
        if st.button(suggestion, key=f"s_{i}", use_container_width=True):
            st.session_state.pending_question = suggestion

# ── CHAT HISTORY ──────────────────────────────────────────────────
st.divider()

for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])

# ── QUESTION INPUT ────────────────────────────────────────────────
pending = st.session_state.get("pending_question", None)
question = st.chat_input("Ask anything about Rithujaa...")

if pending and not question:
    question = pending
    st.session_state.pending_question = None

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(
                question,
                st.session_state.vectorstore,
                st.session_state.chat_history
            )
        st.write(answer)

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer
    })

# ── CLEAR BUTTON ──────────────────────────────────────────────────
if st.session_state.chat_history:
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
This is an AI-powered profile assistant for **Rithujaa Rajendrakumar**.

Built using:
- GPT-4o-mini
- RAG (Retrieval Augmented Generation)
- LangChain + ChromaDB
- Streamlit

Ask anything about her experience, projects, skills, education, or background.
""")
    st.divider()
    st.markdown("### Connect")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/rithujaa/)")
    st.markdown("[GitHub](https://github.com/rithujaa)")
    st.markdown("[Portfolio](https://rithujaa.github.io/rithujaa-portfolio/)")