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

st.set_page_config(page_title="Rithujaa Rajendrakumar", page_icon="👩‍💻", layout="centered")

# ── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default streamlit header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main background */
    .stApp { background-color: #0d1117; }

    /* Avatar */
    .avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: linear-gradient(135deg, #e94560, #0f3460);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: bold;
        color: white;
        margin: 0 auto 16px auto;
        border: 3px solid #e94560;
    }

    /* Header card */
    .profile-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 16px;
        padding: 28px 30px;
        text-align: center;
        margin-bottom: 24px;
    }
    .profile-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        margin: 0 0 4px 0;
        font-weight: 700;
    }
    .profile-header .title {
        color: #e94560;
        font-size: 1rem;
        margin: 0 0 16px 0;
        font-weight: 500;
    }
    .profile-header .bio {
        color: #a8b2d8;
        font-size: 0.88rem;
        margin: 0 0 20px 0;
        line-height: 1.6;
    }
    .links {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .link-btn {
        background: rgba(233, 69, 96, 0.15);
        border: 1px solid #e94560;
        border-radius: 20px;
        padding: 6px 16px;
        color: #e94560 !important;
        text-decoration: none !important;
        font-size: 0.82rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .link-btn:hover {
        background: #e94560;
        color: white !important;
    }

    /* Tags */
    .tags {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .tag {
        background: rgba(15, 52, 96, 0.6);
        border: 1px solid #1f4e8c;
        border-radius: 12px;
        padding: 4px 12px;
        color: #a8b2d8;
        font-size: 0.78rem;
    }

    /* Section headers */
    .section-label {
        color: #a8b2d8;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    /* Suggestion buttons */
    .stButton > button {
        background: #1a1a2e !important;
        border: 1px solid #2d2d44 !important;
        border-radius: 10px !important;
        color: #a8b2d8 !important;
        font-size: 0.83rem !important;
        text-align: left !important;
        padding: 10px 14px !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        border-color: #e94560 !important;
        color: #e94560 !important;
        background: rgba(233, 69, 96, 0.08) !important;
    }

    /* Chat messages */
    .stChatMessage {
        background: #1a1a2e !important;
        border-radius: 12px !important;
        border: 1px solid #2d2d44 !important;
        margin-bottom: 8px !important;
    }

    /* Divider */
    hr { border-color: #2d2d44 !important; }

    /* Sidebar */
    .stSidebar { background: #0d1117 !important; }
    .stSidebar .stMarkdown { color: #a8b2d8 !important; }

    /* RAG badge */
    .rag-badge {
        display: inline-block;
        background: rgba(233, 69, 96, 0.1);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 8px;
        padding: 3px 10px;
        color: #e94560;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_base_built" not in st.session_state:
    st.session_state.knowledge_base_built = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

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
    all_text = ""
    for path in DOCUMENT_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(all_text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
    chroma_client = chromadb.Client()
    vectorstore = Chroma.from_texts(
        chunks, embeddings,
        client=chroma_client,
        collection_name="rithujaa_profile"
    )
    return vectorstore


def answer_question(question, vectorstore, chat_history):
    history_text = ""
    if chat_history:
        history_text = "Previous conversation:\n"
        for item in chat_history[-3:]:
            history_text += f"Q: {item['question']}\nA: {item['answer']}\n\n"

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a helpful assistant that answers questions about Rithujaa Rajendrakumar based on the information provided below.

Answer naturally and conversationally. Be specific and accurate. If the information is not in the context, say you don't have that detail.

{history_text}

Context about Rithujaa:
{context}

Question: {question}

Answer:"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    response = llm.invoke(prompt)
    return response.content.strip(), docs


# ── BUILD KNOWLEDGE BASE ON STARTUP ───────────────────────────────
if not st.session_state.knowledge_base_built:
    with st.spinner("Loading profile..."):
        st.session_state.vectorstore = build_knowledge_base()
        st.session_state.knowledge_base_built = True

# ── PROFILE HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="profile-header">
    <div class="avatar">RR</div>
    <h1>Rithujaa Rajendrakumar</h1>
    <p class="title">MS Data Science, NYU &nbsp;·&nbsp; Data Science &nbsp;·&nbsp; Data Engineering &nbsp;·&nbsp; AI</p>
    <p class="bio">
        NYU MS Data Science grad (100% CDS Pathbreaker Scholarship) with experience in 
        data engineering, machine learning, and AI. Built production pipelines at Sonline LLC 
        and LLM-powered strategies at Trilemma Foundation. Currently building AI applications 
        and actively seeking full-time roles in data and AI.
    </p>
    <div class="tags">
        <span class="tag">🐍 Python</span>
        <span class="tag">🗄️ SQL</span>
        <span class="tag">📊 Power BI & Tableau</span>
        <span class="tag">❄️ Snowflake</span>
        <span class="tag">🤖 LLMs</span>
        <span class="tag">☁️ AWS & Azure</span>
    </div>
    <div class="links">
        <a class="link-btn" href="https://www.linkedin.com/in/rithujaa/" target="_blank">💼 LinkedIn</a>
        <a class="link-btn" href="https://github.com/rithujaa" target="_blank">🐙 GitHub</a>
        <a class="link-btn" href="https://rithujaa.github.io/rithujaa-portfolio/" target="_blank">🌐 Portfolio</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SUGGESTED QUESTIONS ───────────────────────────────────────────
st.markdown('<p class="section-label">💡 Suggested Questions</p>', unsafe_allow_html=True)

suggestions = [
    "🛠️ What tools and skills does she have?",
    "🎓 What is her educational background?",
    "❄️ What is her experience with Snowflake?",
    "🤖 Has she worked with LLMs?",
    "📍 Is she open to relocation?",
    "🎯 What kind of roles is she looking for?",
]

col1, col2 = st.columns(2)
for i, suggestion in enumerate(suggestions):
    with col1 if i % 2 == 0 else col2:
        if st.button(suggestion, key=f"s_{i}", use_container_width=True):
            st.session_state.pending_question = suggestion

st.divider()

# ── CHAT HISTORY ──────────────────────────────────────────────────
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
        with st.spinner("Searching Rithujaa's profile..."):
            answer, _ = answer_question(
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
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👩‍💻 About Rithujaa")
    st.markdown("""
Originally from Sri Lanka. BS in Data Science from UBC, MS from NYU on a full scholarship.

**Currently:** Actively seeking full-time roles in Data Science, Data Engineering, Data Analytics, and AI.

**Open to:** Relocating anywhere in the US.
""")
    st.divider()

    st.markdown("### 🧠 How This Works")
    st.markdown("""
This chatbot uses **RAG (Retrieval Augmented Generation)**:

1. Rithujaa's profile is stored across 6 structured documents
2. Each document is chunked and embedded using OpenAI Embeddings
3. Your question is matched against the most relevant chunks
4. GPT-4o-mini generates an answer from retrieved context

Built with **LangChain + ChromaDB + Streamlit**.
""")
    st.divider()

    st.markdown("### 🔗 Links")
    st.markdown("""
- [LinkedIn](https://www.linkedin.com/in/rithujaa/)
- [GitHub](https://github.com/rithujaa)
- [Portfolio](https://rithujaa.github.io/rithujaa-portfolio/)
""")

    st.divider()
    st.markdown('<span class="rag-badge">⚡ Powered by RAG</span>', unsafe_allow_html=True)