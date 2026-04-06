import streamlit as st
import time

from models import bart_model, t5_model, pegasus_model
from models import gpt2_model, bert_model, lsa_model

st.set_page_config(layout="wide")
st.title("🧠 Multi-Model Text Summarizer")

text = st.text_area("Enter text:", height=250)

max_len = st.slider("Max Length", 30, 200, 100)
min_len = st.slider("Min Length", 10, 100, 30)

if st.button("Generate Summaries"):

    if not text.strip():
        st.warning("Enter text first")
    else:
        results = {}
        times = {}

        models = {
            "BART": lambda: bart_model.summarize(text, max_len, min_len),
            "T5": lambda: t5_model.summarize(text, max_len, min_len),
            "PEGASUS": lambda: pegasus_model.summarize(text, max_len, min_len),
            "GPT-2": lambda: gpt2_model.summarize(text),
            "BERT": lambda: bert_model.summarize(text),
            "LSA": lambda: lsa_model.summarize(text),
        }

        for name, func in models.items():
            start = time.time()
            try:
                results[name] = func()
            except Exception as e:
                results[name] = f"Error: {e}"
            end = time.time()
            times[name] = round(end - start, 2)

        st.subheader("Results")

        cols = st.columns(3)

        for i, (name, summary) in enumerate(results.items()):
            with cols[i % 3]:
                st.markdown(f"### {name}")
                st.write(summary)
                st.caption(f"{times[name]} sec")