from sentence_transformers import SentenceTransformer, util
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


IMDB_data = pd.read_csv("IMDB Dataset.csv")

corpus = IMDB_data['review'][0:10].tolist()
topics = ["football", "economy","politics", "coffee", "watching", "episode", "movie", "film", "like", "good", "bad", "arts", "think", "horror", "action", "story", "theatrer", "filmmakers", "performance", "name"]

corpus.append("This man doing some great work in comunity and help to have better country")
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


doc_embeddings = model.encode(corpus)
topic_embeddings = model.encode(topics)
print(topic_embeddings[0].shape)

# Set similarity threshold
threshold = 0.2

# Create graph
G = nx.Graph()

# Add topics as nodes
for i, topic in enumerate(topics):
    G.add_node(topic, type='topic', bipartite=0)

# Add documents as nodes and connect to topics based on similarity
for i, doc_embedding in enumerate(doc_embeddings):
    G.add_node(f"doc{i}", type='document', bipartite=1)
    for j, topic_embedding in enumerate(topic_embeddings):
        similarity = util.cos_sim(topic_embedding, doc_embedding)
        #print(similarity, j)
        #print(np.linalg.norm(doc_embedding))
        if similarity > threshold:
            G.add_edge(topics[j], f"doc{i}", weight=similarity[0][0].item())

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



