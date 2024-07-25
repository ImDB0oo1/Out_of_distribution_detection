import gensim
from gensim import corpora
from pprint import pprint
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS
import re

def remove_html_tags(text):
    pattern = re.compile("<.*?>")
    return pattern.sub(r"", text)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

import string
exclude = string.punctuation
def remove_punctuation(text):
    for char in exclude:
        text = text.replace(char, "")
    return text
chat_word = {
    'AFAIK': 'As Far As I Know',
    'AFK': 'Away From Keyboard',
    'ASAP': 'As Soon As Possible',
    'ATK': 'At The Keyboard',
    'ATM': 'At The Moment',
    'A3': 'Anytime, Anywhere, Anyplace',
    'BAK': 'Back At Keyboard',
    'BBL': 'Be Back Later',
    'BBS': 'Be Back Soon',
    'BFN': 'Bye For Now',
    'B4N': 'Bye For Now',
    'BRB': 'Be Right Back',
    'BRT': 'Be Right There',
    'BTW': 'By The Way',
    'B4': 'Before',
    'CU': 'See You',
    'CUL8R': 'See You Later',
    'CYA': 'See You',
    'FAQ': 'Frequently Asked Questions',
    'FC': 'Fingers Crossed',
    'FWIW': "For What It's Worth",
    'FYI': 'For Your Information',
    'GAL': 'Get A Life',
    'GG': 'Good Game',
    'GN': 'Good Night',
    'GMTA': 'Great Minds Think Alike',
    'GR8': 'Great!',
    'G9': 'Genius',
    'IC': 'I See',
    'ICQ': 'I Seek you (also a chat program)',
    'ILU': 'ILU: I Love You',
    'IMHO': 'In My Honest/Humble Opinion',
    'IMO': 'In My Opinion',
    'IOW': 'In Other Words',
    'IRL': 'In Real Life',
    'KISS': 'Keep It Simple, Stupid',
    'LDR': 'Long Distance Relationship',
    'LMAO': 'Laugh My A.. Off',
    'LOL': 'Laughing Out Loud',
    'LTNS': 'Long Time No See',
    'L8R': 'Later',
    'MTE': 'My Thoughts Exactly',
    'M8': 'Mate',
    'NRN': 'No Reply Necessary',
    'OIC': 'Oh I See',
    'PITA': 'Pain In The A..',
    'PRT': 'Party',
    'PRW': 'Parents Are Watching',
    'QPSA?': 'Que Pasa?',
    'ROFL': 'Rolling On The Floor Laughing',
    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',
    'ROTFLMAO': 'Rolling On The Floor Laughing My A.. Off',
    'SK8': 'Skate',
    'STATS': 'Your sex and age',
    'ASL': 'Age, Sex, Location',
    'THX': 'Thank You',
    'TTFN': 'Ta-Ta For Now!',
    'TTYL': 'Talk To You Later',
    'U': 'You',
    'U2': 'You Too',
    'U4E': 'Yours For Ever',
    'WB': 'Welcome Back',
    'WTF': 'What The F...',
    'WTG': 'Way To Go!',
    'WUF': 'Where Are You From?',
    'W8': 'Wait...',
    '7K': 'Sick:-D Laugher',
    'TFW': 'That feeling when',
    'MFW': 'My face when',
    'MRW': 'My reaction when',
    'IFYP': 'I feel your pain',
    'TNTL': 'Trying not to laugh',
    'JK': 'Just kidding',
    'IDC': "I don't care",
    'ILY': 'I love you',
    'IMU': 'I miss you',
    'ADIH': 'Another day in hell',
    'ZZZ': 'Sleeping, bored, tired',
    'WYWH': 'Wish you were here',
    'TIME': 'Tears in my eyes',
    'BAE': 'Before anyone else',
    'FIMH': 'Forever in my heart',
    'BSAAW': 'Big smile and a wink',
    'BWL': 'Bursting with laughter',
    'BFF': 'Best friends forever',
    'CSL': "Can't stop laughing"
}

def short_conv(text):
    new_text = []  # Initialize an empty list to hold the processed words
    for w in text.split():  # Split the input text into words and iterate over them
        if w.upper() in chat_word:  # Check if the uppercase version of the word is in the chat_word dictionary
            new_text.append(chat_word[w.upper()])  # If it is, append the full form from the dictionary to new_text
        else:
            new_text.append(w)  # If it is not, append the original word to new_text
    return " ".join(new_text)  # Join the processed words into a single string and return it

from nltk.corpus import stopwords
stop_words = stopwords.words("english")

def remove_stopwords(text):
    new_text=[]
    for word in text.split():
        if word in stop_words:
            new_text.append('')
        else:
            new_text.append(word)

    x = new_text[:]  # Create a copy of new_text
    new_text.clear()  # Clear the original new_text list
    return " ".join(x)  # Join the copied list x into a single string separated by spaces and return it

# Removing Emojis
def remove_emoji(text):
    emoji_pattern=re.compile("["
                             u"\U0001F600-\U0001F64F" #emoticons
                             u"\U0001F300-\U0001F5FF" #symbols, pictograph
                              u"\U0001F680-\U0001F6FF" #transport and map symbol
                              u"\U0001F1E0-\U0001F1FF" #flags(IOS)
                              u"\U00002702-\U000027B0"
                              u"\U00002FC2-\U0001F251"
                             "]+",flags=re.UNICODE)
    return emoji_pattern.sub(r'',text)

# IMDB_data = pd.read_csv("C:/Users/ImDB/Desktop/uni/thesis/IMDB Dataset.csv")
# corpus = IMDB_data["review"]

from sklearn.datasets import fetch_20newsgroups
categories = ['rec.sport.baseball', 
              'rec.sport.hockey']
data_train = fetch_20newsgroups(
    subset='all',
    categories=categories,
    shuffle=True,
    random_state=42
)
IMDB_data = data_train.data
IMDB_data = pd.DataFrame(IMDB_data)
corpus = IMDB_data[0]
print(corpus[0])
# Additional words to remove
additional_stopwords = set(["the", "its", "is", "was", "/><br", "it\'s", "/>the", "<br"])  # Add more if needed

# Function to preprocess a document by removing stopwords
def preprocess_document(doc):
    return [word for word in doc.lower().split() if word not in STOPWORDS.union(additional_stopwords)]

corpus = corpus.str.lower()
corpus = corpus.apply(remove_emoji)
corpus = corpus.apply(remove_stopwords)
corpus = corpus.apply(short_conv)
corpus = corpus.apply(remove_punctuation)
corpus = corpus.apply(remove_url)
corpus = corpus.apply(remove_html_tags)

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






