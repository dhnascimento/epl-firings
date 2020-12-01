
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import string

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

def lemmaNLTK(sentence):
    # Assigning lemmatizer to variable
    lemmatizer = WordNetLemmatizer()
    # Tokenizing the words
    listofwords = re.split(" |'",sentence)
    # List for storing the words after lowercasing and removing punctuation
    listofwords2 = []
    listoflemma_words = []
    # List of punctuations
    table = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation} 
    
    
    for word in listofwords:
        # Removing punctuaion
        word_p = word.translate(table)
        # Lower case
        word_l = word_p.lower()
        # Appending to list for next loop
        listofwords2.append(word_l)
    
    for word_l in listofwords2:
        # Ignore words in STOPLIST
        if word_l in STOPLIST: continue      
        # Lemmatizing the words
        lemma_word = lemmatizer.lemmatize(word_l)
        # Appending lemmatized words to list
        listoflemma_words.append(lemma_word)

    # Removing empty entries from the list    
    listoflemma_words = list(filter(None, listoflemma_words))
        
    return listoflemma_words