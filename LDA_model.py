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
num_topics = 20

# Build LDA model
lda_model = gensim.models.LdaModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=num_topics, random_state=42)
#filtered_topics = [(topic_number, [word for word, prob in lda_model.show_topic(topic_number) if word not in STOPWORDS]) for topic_number in range(num_topics)]
# Print topics and associated keywords
print("Topics and Keywords:")
pprint(lda_model.print_topics())





import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


topics = ["acting", "plot", "cinematography", "characters", "dialogue", "overall"]

# Preprocess documents and topics, and generate BERT embeddings
def get_bert_embeddings(text):
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs[0][:, 0, :].numpy()
    return embeddings

doc_embeddings = [get_bert_embeddings(doc) for doc in corpus]
topic_embeddings = [get_bert_embeddings(topic) for topic in topics]

# Calculate cosine similarity between topic embeddings and document embeddings
def calculate_cosine_similarity(topic_embedding, doc_embedding):
    similarity = cosine_similarity(topic_embedding, doc_embedding)
    return similarity[0][0]

# Set similarity threshold
threshold = 0.5

# Create graph
G = nx.Graph()

# Add topics as nodes
for i, topic in enumerate(topics):
    G.add_node(topic, type='topic')

# Add documents as nodes and connect to topics based on similarity
for i, doc_embedding in enumerate(doc_embeddings):
    G.add_node(f"doc{i}", type='document')
    for j, topic_embedding in enumerate(topic_embeddings):
        similarity = calculate_cosine_similarity(topic_embedding, doc_embedding)
        if similarity > threshold:
            G.add_edge(topics[j], f"doc{i}", weight=similarity)

# Plot graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=5000, node_color='lightblue', font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
