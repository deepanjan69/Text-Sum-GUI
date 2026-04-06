from transformers import pipeline

model = pipeline("summarization", model="google/pegasus-xsum")

def summarize(text, max_len, min_len):
    return model(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']