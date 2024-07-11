import torch
from transformers import BertTokenizer, BertModel, LongformerTokenizer, LongformerModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load pre-trained BERT model and tokenizer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
model.eval()

# Load corpus
IMDB_data = pd.read_csv("C:/Users/ImDB/Desktop/uni/thesis/IMDB Dataset.csv")
#concatenated_text = ' '.join(IMDB_data['review'][0:50])
corpus = IMDB_data["review"][0:2]
topics = ["watching", "episode", "movie", "film", "like", "good", "bad", "arts", "think", "horror", "action", "story", "theatrer", "filmmakers", "performance", "name"]

# Preprocess documents and topics, and generate BERT embeddings
def get_bert_embeddings(text, tokenizer):
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])    
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs[0][:, 0, :].numpy()
    return embeddings

doc_embeddings = [get_bert_embeddings(doc, tokenizer) for doc in corpus]
topic_embeddings = [get_bert_embeddings(topic, tokenizer) for topic in topics]
# # Save embeddings to a file
# np.save('doc_embeddings.npy', doc_embeddings)
# np.save('topic_embeddings.npy', topic_embeddings)
#doc_embeddings = np.load('doc_embeddings.npy')
#topic_embeddings = np.load('topic_embeddings.npy')



# Normilize embeddings 
doc_embeddings = [doc/np.linalg.norm(doc) for doc in doc_embeddings]
topic_embeddings = [topic/np.linalg.norm(topic) for topic in topic_embeddings]

# norms = np.linalg.norm(doc_embeddings, axis=0, keepdims=True)
# doc_embeddings = doc_embeddings / norms
# norms = np.linalg.norm(topic_embeddings, axis=0, keepdims=True)
# topic_embeddings = topic_embeddings / norms

# Calculate cosine similarity between topic embeddings and document embeddings
def calculate_cosine_similarity(topic_embedding, doc_embedding):
    similarity = cosine_similarity(topic_embedding, doc_embedding)
    return similarity[0][0]

# Set similarity threshold
threshold = 0.46

# Create graph
G = nx.Graph()

# Add topics as nodes
for i, topic in enumerate(topics):
    G.add_node(topic, type='topic', bipartite=0)

# Add documents as nodes and connect to topics based on similarity
for i, doc_embedding in enumerate(doc_embeddings):
    G.add_node(f"doc{i}", type='document', bipartite=1)
    for j, topic_embedding in enumerate(topic_embeddings):
        similarity = calculate_cosine_similarity(topic_embedding, doc_embedding)
        print(similarity, i)
        print(np.linalg.norm(doc_embedding))
        if similarity > threshold:
            G.add_edge(topics[j], f"doc{i}", weight=similarity)

# Plot graph
plt.figure(figsize=(10,8))
pos = nx.bipartite_layout(G, topics)
#nx.draw_networkx_nodes(G, pos, nodelist=topics, node_color='r', label="topics")
#nx.draw_networkx_nodes(G, pos, nodelist=doc_embeddings, node_color='b', label="document")
nx.draw(G, pos, with_labels=True, node_size=500, font_size=5, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size = 8)
plt.title('Graph of topics and documents')
plt.show()

