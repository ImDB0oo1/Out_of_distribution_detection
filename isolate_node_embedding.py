from sentence_transformers import SentenceTransformer, util
import numpy as np



sbert_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
embedding = sbert_model.encode("These are isolate nodes")

np.save('isolate_node.npy', embedding)

akbar = np.load('isolate_node.npy')
print(akbar)