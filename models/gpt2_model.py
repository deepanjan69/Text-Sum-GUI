from transformers import pipeline

model = pipeline("text-generation", model="gpt2")

def summarize(text):
    output = model("Summarize: " + text, max_length=120)
    return output[0]['generated_text']