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

# Load pre-trained BERT model and tokenizer
word_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
word_model = BertModel.from_pretrained('bert-base-uncased')
word_model.eval()

# Preprocess documents and topics, and generate BERT embeddings
def get_bert_embeddings(text, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])    
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs[0][:, 0, :].numpy()
    return embeddings

doc = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."

topic = "watching"

doc_embedding = get_bert_embeddings(doc, tokenizer, model)
topic_embedding = get_bert_embeddings(topic, word_tokenizer, word_model)
print(doc_embedding.shape)
print(topic_embedding.shape)

# Calculate cosine similarity between topic embeddings and document embeddings
def calculate_cosine_similarity(topic_embedding, doc_embedding):
    similarity = cosine_similarity(topic_embedding, doc_embedding)
    return similarity[0][0]


# Normilize embeddings 
norms = np.linalg.norm(doc_embedding, axis =1, keepdims=True)
doc_embedding = doc_embedding / norms

norms = np.linalg.norm(topic_embedding, axis=1, keepdims=True)
topic_embedding = topic_embedding / norms
print(np.linalg.norm(doc_embedding))
print(np.linalg.norm(topic_embedding))
print(cosine_similarity(doc_embedding, topic_embedding))