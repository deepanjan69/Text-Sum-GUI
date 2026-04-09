import streamlit as st
import time

from models import bart_model, t5_model, pegasus_model
from models import gpt2_model, bert_model, lsa_model

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt')

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(layout="wide")
st.title("🧠 Multi-Model Text Summarizer")

# -----------------------------
# INPUT
# -----------------------------
text = st.text_area("Enter text:", height=250)

max_len = st.slider("Max Length", 30, 200, 100)
min_len = st.slider("Min Length", 10, 100, 30)

# -----------------------------
# GENERATE
# -----------------------------
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

        # -----------------------------
        # RUN MODELS
        # -----------------------------
        for name, func in models.items():
            start = time.time()
            try:
                results[name] = func()
            except Exception as e:
                results[name] = f"Error: {e}"
            end = time.time()
            times[name] = round(end - start, 2)

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        st.subheader("📊 Model Outputs")

        cols = st.columns(3)

        for i, (name, summary) in enumerate(results.items()):
            with cols[i % 3]:
                st.markdown(f"### {name}")
                st.write(summary)
                st.caption(f"⏱ {times[name]} sec")

        # -----------------------------
        # EVALUATION METRICS
        # -----------------------------
        st.subheader("📈 Accuracy Rankings")

        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )

        base = list(results.values())[0]

        evaluation_data = []

        for name, summary in results.items():
            try:
                rouge_scores = scorer.score(base, summary)

                rouge1 = rouge_scores['rouge1'].fmeasure
                rouge2 = rouge_scores['rouge2'].fmeasure
                rougeL = rouge_scores['rougeL'].fmeasure

                bleu = sentence_bleu([base.split()], summary.split())

            except:
                rouge1, rouge2, rougeL, bleu = 0, 0, 0, 0

            evaluation_data.append({
                "Model": name,
                "ROUGE-1": round(rouge1, 3),
                "ROUGE-2": round(rouge2, 3),
                "ROUGE-L": round(rougeL, 3),
                "BLEU": round(bleu, 3)
            })

        df = pd.DataFrame(evaluation_data)

        st.dataframe(df)

        # -----------------------------
        # BEST MODEL
        # -----------------------------
        df["Score"] = df[["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]].mean(axis=1)
        best_model = df.sort_values("Score", ascending=False).iloc[0]

        st.success(
            f"🏆 Best Model: {best_model['Model']} (Score: {round(best_model['Score'],3)})"
        )

        # -----------------------------
        # CHANGE % FROM ORIGINAL TEXT
        # -----------------------------
        st.subheader("📊 Change from Original Text")

        scorer_change = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

        change_data = {}

        for name, summary in results.items():
            try:
                score = scorer_change.score(text, summary)
                similarity = score['rouge1'].fmeasure

                change_percent = (1 - similarity) * 100

            except:
                change_percent = 0

            change_data[name] = round(change_percent, 2)

        # Show values
        st.write("### Change % per Model")
        st.write(change_data)

        # -----------------------------
        # PIE CHART (FINAL FEATURE)
        # -----------------------------
        st.write("### 🥧 Change Distribution (Based on Original Text)")

        labels = list(change_data.keys())
        values = list(change_data.values())

        fig, ax = plt.subplots(figsize=(3.15, 3.15))  # 8cm × 8cm

        ax.pie(values, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 8})
        ax.set_title("Change Distribution", fontsize=10)

        st.pyplot(fig, use_container_width=False)