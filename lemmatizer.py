import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import string

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
        self.table = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation}

    def __call__(self, sentence):
        # Tokenizing the words
        listofwords = re.split(" |'",sentence)
        # List for storing the words after lowercasing and removing punctuation
        listofwords2 = []
        listoflemma_words = []

        for word in listofwords:
            # Removing punctuation
            word_p = word.translate(self.table)
            # Lower case
            word_l = word_p.lower()
            # Appending to list for next loop
            listofwords2.append(word_l)

        for word_l in listofwords2:
            # Ignore words in STOPLIST
            if word_l in self.STOPLIST: continue
            # Lemmatizing the words
            lemma_word = self.wnl.lemmatize(word_l)
            # Appending lemmatized words to list
            listoflemma_words.append(lemma_word)

        # Removing empty entries from the list
        listoflemma_words = list(filter(None, listoflemma_words))

        return listoflemma_words