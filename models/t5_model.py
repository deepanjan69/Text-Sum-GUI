from transformers import pipeline

model = pipeline("summarization", model="t5-small")

def summarize(text, max_len, min_len):
    return model("summarize: " + text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']