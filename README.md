"# Out_of_distribution_detection" 
This repo is about detection OOD sample on natural language proccesing.The model first tries to extract some keywords from main domain text dataset,Then make a bipartite graph between each document and extracted keywords based on their cosine similarity.Finally apply a two-hop gnn model to learn to label some document is in our main domain or not.
