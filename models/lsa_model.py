import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

def summarize(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    lsa = LsaSummarizer()
    summary = lsa(parser.document, 3)
    return " ".join([str(s) for s in summary])