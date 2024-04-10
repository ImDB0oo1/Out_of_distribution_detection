import gensim
from gensim import corpora
from pprint import pprint
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS

IMDB_data = pd.read_csv("C:/Users/ImDB/Desktop/uni/thesis/IMDB Dataset.csv")
concatenated_text = ' '.join(IMDB_data['review'][0:100])
corpus = IMDB_data["review"]


# Additional words to remove
additional_stopwords = set(["the", "its", "is", "was", "/><br", "it\'s", "/>the", "<br"])  # Add more if needed

# Function to preprocess a document by removing stopwords
def preprocess_document(doc):
    return [word for word in doc.lower().split() if word not in STOPWORDS.union(additional_stopwords)]

# Preprocess corpus
preprocessed_corpus = [preprocess_document(doc) for doc in corpus]

# Tokenize documents
tokenized_corpus = [doc.lower().split() for doc in corpus if doc not in STOPWORDS]

# Create dictionary
dictionary = corpora.Dictionary(preprocessed_corpus)

# Create document-term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokenized_corpus]

# Define number of topics
num_topics = 10

# Build LDA model
lda_model = gensim.models.LdaModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=num_topics, random_state=42)
#filtered_topics = [(topic_number, [word for word, prob in lda_model.show_topic(topic_number) if word not in STOPWORDS]) for topic_number in range(num_topics)]
# Print topics and associated keywords
print("Topics and Keywords:")
pprint(lda_model.print_topics())






