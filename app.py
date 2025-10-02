# app.py
import os
import streamlit as st
from src.summarizer import summarize_long_text
import re

st.set_page_config(page_title="Medical Text Summarizer", page_icon="🩺", layout="wide")
st.title("🩺 Medical Text Summarizer")

with st.sidebar:
    st.header("Settings")
    default_model = os.getenv("MODEL_SOURCE", "facebook/bart-large-cnn")
    model_source = st.text_input("Model source (HF id or local path)", value=default_model)
    use_gpu = st.checkbox("Use GPU (if available)", value=False)
    min_len = st.slider("Min summary length (tokens)", 10, 200, 40)
    max_len = st.slider("Max summary length (tokens)", 40, 400, 150)
    second_pass = st.checkbox("Second-pass summary (shorten stitched parts)", value=True)
    deid_toggle = st.checkbox("Light de-identify input before summarizing (heuristic)", value=False)
    private_token = st.text_input("HF token (for private model)", type="password")

def deidentify(text: str) -> str:
    text = re.sub(r"\b\d{2,4}[-/]\d{1,2}[-/]\d{1,2}\b", "[DATE]", text)
    text = re.sub(r"\b\d{10,15}\b", "[ID]", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)
    return text

st.markdown("---")
st.markdown("Paste a clinical note or upload a `.txt` file.")

col1, col2 = st.columns([3,1])

with col1:
    text_input = st.text_area("Clinical note (paste here)", height=350)
    uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])
    if uploaded is not None:
        bytes_data = uploaded.read()
        try:
            text_input = bytes_data.decode("utf-8")
        except:
            text_input = bytes_data.decode("latin-1")

with col2:
    st.write("**How to use**")
    st.write("- Paste notes or upload a text file.")
    st.write("- Adjust min/max length.")
    st.write("- If your model is private, paste HF token in sidebar.")

if st.button("Summarize"):
    if not text_input or not text_input.strip():
        st.warning("Please paste or upload a clinical note first.")
    else:
        input_text = text_input
        if deid_toggle:
            input_text = deidentify(input_text)

        with st.spinner("Summarizing (this may take a while on first load)..."):
            summary = summarize_long_text(
                input_text,
                model_source=model_source,
                use_gpu=use_gpu,
                min_length=min_len,
                max_length=max_len,
                second_pass=second_pass,
                use_auth_token=private_token if private_token else None
            )

        st.subheader("📝 Summary")
        st.write(summary)
        st.download_button("Download summary as .txt", summary, file_name="summary.txt")

st.markdown("---")
st.caption("Note: Demo for educational use only; not for clinical diagnosis.")
