import nltk
from nltk.tokenize import PunktSentenceTokenizer

# Explicitly specify the nltk_data directory
nltk.data.path.append('/home/atmri/nltk_data')

# Load tokenizer directly from standard punkt
tokenizer = PunktSentenceTokenizer()
print(tokenizer.tokenize("Testing NLTK after cleaning up 'punkt_tab'."))
