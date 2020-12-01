import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.utils import resample
import re
import string

import pickle

# Importing resample 
# logit = load('epl_logit.joblib')

df_articles = pd.read_csv('capstone_final_dataset.csv', index_col=False)
df_articles.drop('Unnamed: 0', axis=1, inplace=True)

# Lemmatizer function with NLTK

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


# Defining the training and test sets to test a model with the original unbalanced data
X_train, X_test, y_train, y_test = train_test_split(df_articles['Articles'], df_articles['Status'], test_size = 0.3, 
random_state =42, stratify = df_articles['Status'] )

                                                    
# Defining the training and validation sets to test a model with the original unbalanced data
X_train_p, X_val, y_train_p, y_val = train_test_split(X_train, y_train, test_size = 0.2, 
random_state =42, stratify = y_train )

# Creating a DataFrame to proceed with the rebalance

df_bal = pd.DataFrame({'Articles': X_train_p, 'Status': y_train_p})

# Separating the balanced class (0) from the imbalanced class (1)

imbal_class = df_bal[df_bal['Status']==1]
bal_class = df_bal[df_bal['Status']==0]

# Applying the resample function to oversample class 1

imbal =resample(imbal_class, random_state = 42, n_samples = 1944)

# Merging the resampled class to create a balanced data set

overs_train = pd.concat([bal_class, imbal])

# Getting the X and y matrices to train the models

X_train_ov = overs_train['Articles']
y_train_ov = overs_train['Status']


# Instantiating and fitting the TF-IDF model to the resampled training set
tfidf = TfidfVectorizer(min_df=1, tokenizer=lemmaNLTK).fit(X_train_ov)

# Transforming train, validation and test sets in document-term matrices
X_train_tf_df1 = tfidf.transform(X_train_ov)
X_val_tf_df1 = tfidf.transform(X_val)
X_test_tf_df1 = tfidf.transform(X_test)

# Instantiating and fitting the model
# logit = LogisticRegression(solver='saga', C=100, penalty='l2').fit(X_train_tf_df1, y_train_ov)

# # Calculating the scores for the train, validation and test sets
# print(f'Train set score: {logit.score(X_train_tf_df1, y_train_ov)}')
# print(f'Validation set score: {logit.score(X_val_tf_df1, y_val)}')
# print(f'Test set score: {logit.score(X_test_tf_df1, y_test)}')

#Create pipeline 
pipeline = Pipeline([('vector', tfidf), ('model', LogisticRegression(solver='saga', C=100, penalty='l2'))])

pipeline.fit(X_train_ov, y_train_ov)

#Export pipeline
with open('model.pkl', 'wb') as model_file:
  pickle.dump(pipeline, model_file)


#Load and test
model = pickle.load(open('model.pkl', 'rb'))

print(model.predict(["Mauricio Pochettino is heading towards the point of no return at Tottenham Hotspur as fears grow that he will not be able to salvage their season. The West Ham United game immediately after the international break is now rated as make or break for Tottenham’s campaign, and could even prove decisive for the manager himself. The home draw with Sheffield United last Saturday left Spurs 14th in the Premier League going into the international break, just a point ahead of West Ham and without a win in five league games. Tottenham trail fourth-placed Manchester City by 11 points. West Ham have not won in the league since September, but Spurs have not won away from home in the league since January, when they beat Fulham. Another defeat on the road, at the home of their fierce rivals, would raise serious questions for the Spurs’ manager. Pochettino is understood to have been at Tottenham’s Enfield training centre this week as he attempts to turn around the club’s season. He has already warned there is no quick fix and last Saturday appeared to acknowledge that he may not get the time to make the changes he wants, saying: “We are in the process to build and we will see if we have the time to build what we want.” West Ham and Arsenal have this week put out messages of support for their under-pressure managers, but there has been no such public backing for Pochettino. That could be because Spurs are reluctant to be seen to be giving a dreaded vote of confidence, but the silence has only strengthened theories that a change is becoming inevitable. It could also be a consequence of the fact that Pochettino has given the impression he could eventually walk away after first floating the idea ahead of last season’s Champions League final.  Pochettino signed a five-year contract worth up to £8.5 million a year last May, with sources claiming it would cost chairman Daniel Levy around £12.5 million to sack him. Spurs have never confirmed the details of Pochettino’s contract, but Levy would prefer not to pay an expensive compensation bill to sack a manager who has done so much for the club. There is also the issue over who would replace Pochettino midseason, even in a caretaker position, with no obvious interim at the club. That would enhance the prospects of an out-of-work manager, such as Max Allegri or Jose Mourinho, being approached. But waiting to the end of the season would widen Levy’s options and give him a more realistic chance of appointing Bournemouth’s Eddie Howe, RB Leipzig’s Julian Nagelsmann, or even England manager Gareth Southgate. Pochettino will have no shortage of suitors when he eventually leaves. He is still rated as one of the best in the business and has admirers at Manchester United and Real Madrid. The 47-year-old has worked sporting miracles since being appointed in 2014, taking the club into the Champions League and reaching the final, as well as competing for the Premier League title. Pochettino wanted to make big changes this summer, but Tottenham failed to sell the likes of Christian Eriksen, Toby Alderweireld and Danny Rose, while making only three new signings. Eriksen has denied that he has been dropped as punishment for not signing a new contract, saying: “I feel 100 per cent that Tottenham has complete confidence in me. There is not that big a difference, except that I play a little less this year.“‘I don’t feel there is a connection between my contract situation and the fact that I haven’t played as many matches.”"]))

print("All done!")

