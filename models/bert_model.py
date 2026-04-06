from summarizer import Summarizer

model = Summarizer()

def summarize(text):
    return model(text, num_sentences=3)