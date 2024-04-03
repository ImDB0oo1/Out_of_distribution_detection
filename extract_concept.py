import spacy
from collections import Counter
import pandas as pd

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Sample corpus text data (replace this with your actual corpus)
# corpus_text = """
# Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. It involves the development of algorithms and models that enable computers to understand, interpret, and generate human language. NLP techniques are used in various applications, including machine translation, sentiment analysis, and text summarization. One popular approach in NLP is the use of recurrent neural networks (RNNs), which are particularly effective for sequence modeling tasks.
# """
IMDB_data = pd.read_csv("C:/Users/ImDB/Desktop/uni/thesis/IMDB Dataset.csv")
concatenated_text = ' '.join(IMDB_data['review'][0:100])
#print(concatenated_text)
corpus_text = concatenated_text
# Process the corpus text using spaCy
doc = nlp(corpus_text)

# Extract key concepts using spaCy's named entity recognition (NER)
key_concepts = [ent.text for ent in doc.ents if ent.label_ == "ORG" or ent.label_ == "PERSON" or ent.label_ == "GPE"]

# Alternatively, you can use noun chunks for extracting key concepts
# key_concepts = [chunk.text for chunk in doc.noun_chunks]

# Count the frequency of each key concept
concept_freq = Counter(key_concepts)

# Print the top key concepts
print("Top Key Concepts:")
for concept, freq in concept_freq.most_common(5):
    print(f"{concept}: {freq} occurrences")
