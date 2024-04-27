from sentence_transformers import SentenceTransformer, util
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups


categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space'
]

data_train = fetch_20newsgroups(
    subset='train', 
    categories=categories, 
    shuffle=True, 
    random_state=42
)
OOD_data = data_train.data


IMDB_data = pd.read_csv("IMDB Dataset.csv")

corpus = IMDB_data['review'][0:50].tolist()
topics = ["watching", "episode", "movie", "film", "like", "good", "bad", "arts", "think", "horror", "action", "story", "theatrer", "filmmakers", "performance", "name"]

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

doc_embeddings = model.encode(corpus)
OOD_embeddings = model.encode(OOD_data[0:20])
topic_embeddings = model.encode(topics)
print(OOD_embeddings[0].shape)
print(topic_embeddings[0].shape)


# Set similarity threshold
threshold = 0.17

# Create graph
G = nx.Graph()

# Add topics as nodes
for i, topic_embedding in enumerate(topic_embeddings):
    G.add_node(topics[i], type='topic', bipartite=0, embedding=np.array2string(topic_embedding, separator=',').strip('[]')) 
    print(len(np.array2string(topic_embedding, separator=',').strip('[]')))
# Add documents as nodes and connect to topics based on similarity
for i, doc_embedding in enumerate(doc_embeddings):
    G.add_node(f"ID_doc{i}", type='ID document', bipartite=1, label=0, embedding=np.array2string(doc_embedding, separator=',').strip('[]'))
    print(len(np.array2string(doc_embedding, separator=',').strip('[]')))
    for j, topic_embedding in enumerate(topic_embeddings):
        similarity = util.cos_sim(topic_embedding, doc_embedding)
        #print(similarity, j)
        #print(np.linalg.norm(doc_embedding))
        if similarity > threshold:
            G.add_edge(topics[j], f"ID_doc{i}", weight=similarity[0][0].item())

# Add documents as nodes and connect to topics based on similarity
for i, doc_embedding in enumerate(OOD_embeddings):
    G.add_node(f"OOD_doc{i}", type='OOD document', bipartite=1, label=1, embedding=np.array2string(doc_embedding, separator=',').strip('[]'))
    for j, topic_embedding in enumerate(topic_embeddings):
        similarity = util.cos_sim(topic_embedding, doc_embedding)
        #print(similarity, j)
        #print(np.linalg.norm(doc_embedding))
        if similarity > threshold:
            G.add_edge(topics[j], f"OOD_doc{i}", weight=similarity[0][0].item())

#nx.write_graphml(G, '50ID_20OOD.graphml')

# Plot graph
plt.figure(figsize=(10,8))
pos = nx.bipartite_layout(G, topics)
#nx.draw_networkx_nodes(G, pos, nodelist=topics, node_color='r', label="topics")
#nx.draw_networkx_nodes(G, pos, nodelist=doc_embeddings, node_color='b', label="document")
nx.draw(G, pos, with_labels=True, node_size=500, font_size=5, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size = 8)
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels=labels)
plt.title('Graph of topics and documents')
plt.show()



