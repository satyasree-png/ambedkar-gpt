import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Ambedkar GPT", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size: 32px;
    font-weight: bold;
    color: #2563eb;
}
.card {
    padding: 15px;
    border-radius: 12px;
    background-color: #f1f5f9;
    margin-bottom: 15px;
}
.answer {
    font-size: 16px;
    color: #111827;
}
.highlight {
    background-color: yellow;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">📘 Ambedkar GPT</div>', unsafe_allow_html=True)
st.caption("AI-powered assistant based on Ambedkar's writings")

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_data():
    loader = PyPDFLoader("ambedkar.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    texts = [doc.page_content for doc in chunks]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    return texts, vectorizer, vectors

texts, vectorizer, vectors = load_data()

# ---------------- INPUT ----------------
st.markdown("### 🔍 Ask your question")
query = st.text_input("", placeholder="e.g. What is caste system?")
search = st.button("🚀 Get Answer")

# ---------------- LOGIC ----------------
if search:
    if query.strip() == "":
        st.warning("⚠️ Please enter a question")
    else:
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, vectors)[0]
        top_indices = scores.argsort()[-3:][::-1]

        context = "\n\n".join([texts[i] for i in top_indices])
        clean_text = " ".join(context.split())

        # -------- SENTENCE HIGHLIGHT --------
        sentences = re.split(r'(?<=[.!?]) +', clean_text)

        sentence_scores = []
        for sent in sentences:
            sent_vec = vectorizer.transform([sent])
            score = cosine_similarity(query_vec, sent_vec)[0][0]
            sentence_scores.append(score)

        best_index = sentence_scores.index(max(sentence_scores))
        best_sentence = sentences[best_index]

        highlighted_text = clean_text.replace(
            best_sentence,
            f"<span class='highlight'>{best_sentence}</span>"
        )

        # ---------------- ANSWER ----------------
        st.markdown("### 📘 Answer")
        st.markdown(f"""
        <div class="card answer">
        {highlighted_text[:800]}...
        </div>
        """, unsafe_allow_html=True)

        # ---------------- EXPLANATION ----------------
        st.markdown("### 🧠 Explanation")
        st.markdown("""
        <div class="card">
        This passage discusses the caste system and its impact on society. 
        Ambedkar critically examined caste as a system that enforces inequality 
        and restricts social mobility.
        </div>
        """, unsafe_allow_html=True)

        # ---------------- SOURCES ----------------
        st.markdown("### 📚 Source Chunks")
        for rank, i in enumerate(top_indices):
            st.markdown(f"""
            <div class="card">
            🔹 <b>Rank {rank+1}</b> → Chunk number: {i}
            </div>
            """, unsafe_allow_html=True)