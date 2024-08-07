from sentence_transformers import SentenceTransformer, util
import networkx as nx
import numpy as np
import mathplotlib as plt


#topics = ["watching", "episode", "movie", "film", "like", "good", "bad", "arts", "think", "horror", "action", "story", "theatrer", "filmmakers", "performance", "name"]
topics = ["watching", "episode", "movie", "film", "arts", "story", "theatrer", "filmmakers", "name"]

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

doc_embeddings =  np.load('embeddings.npy')
OOD_embeddings = np.load('embeddings_OOD.npy')
topic_embeddings = model.encode(topics)
print(OOD_embeddings.shape)
print(doc_embeddings.shape)

def make_graph(doc_embeddings, OOD_embeddings, topic_embeddings, knn_mode=False, threshold=0.17, k=4, plot_graph=False):
    G = nx.Graph()
    if knn_mode:
        # Add topics as nodes
        for i, topic_embedding in enumerate(topic_embeddings):
            G.add_node(topics[i], type='topic', bipartite=0, embedding=np.array2string(topic_embedding, separator=',').strip('[]')) 

        # Loop through each document
        for i, doc_embedding in enumerate(doc_embeddings):
            # Add document node to the graph
            G.add_node(f"ID_doc{i}", type='ID document', bipartite=1, label=0, embedding=np.array2string(doc_embedding, separator=',').strip('[]'))
            
            # Calculate cosine similarity between the document and all topics
            similarities = []
            for j, topic_embedding in enumerate(topic_embeddings):
                similarity = util.cos_sim(topic_embedding, doc_embedding)
                similarities.append((similarity[0][0].item(), j))
            
            # Sort the similarities in descending order
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            # Select the top k similarities
            top_4_similarities = similarities[:k]
            
            # Add edges based on the top k similarities
            for similarity, j in top_k_similarities:
                G.add_edge(topics[j], f"ID_doc{i}", weight=similarity)

        # Loop through each OOD document
        for i, doc_embedding in enumerate(OOD_embeddings):
            # Add OOD document node to the graph
            G.add_node(f"OOD_doc{i}", type='OOD document', bipartite=1, label=1, embedding=np.array2string(doc_embedding, separator=',').strip('[]'))
            
            # Calculate cosine similarity between the OOD document and all topics
            similarities = []
            for j, topic_embedding in enumerate(topic_embeddings):
                similarity = util.cos_sim(topic_embedding, doc_embedding)
                similarities.append((similarity[0][0].item(), j))
            
            # Sort the similarities in descending order
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            # Select the top k similarities
            top_k_similarities = similarities[:k]
            
            # Add edges based on the top k similarities
            for similarity, j in top_k_similarities:
                G.add_edge(topics[j], f"OOD_doc{i}", weight=similarity)

        else:            
            # Add documents as nodes and connect to topics based on similarity
            for i, doc_embedding in enumerate(doc_embeddings):
                G.add_node(f"ID_doc{i}", type='ID document', bipartite=1, label=0, embedding=np.array2string(doc_embedding, separator=',').strip('[]'))
                for j, topic_embedding in enumerate(topic_embeddings):
                    similarity = util.cos_sim(topic_embedding, doc_embedding)
                    if similarity > threshold:
                        G.add_edge(topics[j], f"ID_doc{i}", weight=similarity[0][0].item())

            # Add documents as nodes and connect to topics based on similarity
            for i, doc_embedding in enumerate(OOD_embeddings):
                G.add_node(f"OOD_doc{i}", type='OOD document', bipartite=1, label=1, embedding=np.array2string(doc_embedding, separator=',').strip('[]'))
                for j, topic_embedding in enumerate(topic_embeddings):
                    similarity = util.cos_sim(topic_embedding, doc_embedding)
                    if similarity > threshold:
                        G.add_edge(topics[j], f"OOD_doc{i}", weight=similarity[0][0].item())
        # Get the lengths
    doc_embedding_length = len(doc_embeddings)
    ood_embeddings_length = len(OOD_embeddings)
    filename = f"{doc_embedding_length}ID_{ood_embeddings_length}OOD.graphml"
    nx.write_graphml(G, filename)
    if plot_graph:
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



