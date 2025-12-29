import streamlit as st
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

st.title("Resumé Sentiment Analysis")
st.caption("Checking the sentiment and language tone of your resume")

# Input text
text = st.text_area("Enter your resume text here")

if not text.strip():
    st.stop()

# 1. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = [Document(page_content=text)]
chunks = text_splitter.split_documents(documents)

chunk_texts = [chunk.page_content for chunk in chunks]

st.write(f"Created {len(chunk_texts)} chunks.")

# 2. Load pipelines (cached by HF internally)
sentiment_pipeline = pipeline("sentiment-analysis")
tone_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    candidate_labels=[
        "Senior",
        "Junior",
        "Trainee",
        "Blue-collar",
        "White-collar",
        "Self-employed"
    ]
)

# 3. Analyze
if st.button("Analyze"):
    col1, col2 = st.columns(2)

    with col1:
        sentiments = sentiment_pipeline(chunk_texts)

        # Aggregate sentiment by average score
        avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
        label = max(set(s["label"] for s in sentiments),
                    key=lambda x: sum(1 for s in sentiments if s["label"] == x))

        st.subheader("Sentiment")
        st.write(f"Overall Sentiment: **{label}**")
        st.write(f"Confidence: **{avg_score * 100:.1f}%**")

    with col2:
        tones = tone_pipeline(chunk_texts)

        # Aggregate top tone across chunks
        tone_scores = {}
        for result in tones:
            label = result["labels"][0]
            score = result["scores"][0]
            tone_scores[label] = tone_scores.get(label, 0) + score

        best_tone = max(tone_scores, key=tone_scores.get)
        confidence = tone_scores[best_tone] / len(tones)

        st.subheader("Tone")
        st.write(f"Detected Tone: **{best_tone}**")
        st.write(f"Confidence: **{confidence * 100:.1f}%**")
