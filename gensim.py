import logging
import pandas as pd
# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from nltk import download

# Initialize logging.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


IMDB_data = pd.read_csv('C:/Users/ImDB/Desktop/uni/thesis/IMDB Dataset.csv')
corpus = IMDB_data['review'][0:20]

download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

corpus = [preprocess(doc) for doc in corpus]

from gensim.corpora import Dictionary
dictionary = Dictionary(corpus)


dictionary = [dictionary.doc2bow(doc) for doc in corpus]

from gensim.models import TfidfModel
tfidf = TfidfModel(corpus)


doc_tfidf =[tfidf[doc] for doc in dictionary]
