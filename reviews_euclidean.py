# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2nd model: classify according to euclidian
# # # # # # # # # # # # # # # # # # # # # # # # # # #

from math import log
from traceback import print_tb
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import json
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances


# # # #
# 1. Create dataframe from .txt
# # # #

d = { "Labels" : [], "Features" : [] }

with open('train.txt') as f:
    lines = f.readlines()

for line in lines:
    label = line.split()[0]
    d["Labels"].append(label)
    features = line.split()[1:]
    s = ""
    for feature in features:
        s += feature + " "
    s = s[:-1]
    d["Features"].append(s)
    
# # # #
# 2. Process dataset
# # # #
# 2.1 Lowercasing
d["Features"] = [feature.lower() for feature in d["Features"]]

# 2.2 Tokenization
d["Features"] = [nltk.word_tokenize(feature) for feature in d["Features"]]

# 2.3 Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
d["Features"] = [ [wordnet_lemmatizer.lemmatize(word) for word in feature] for feature in d["Features"]]

# 2.4 Stop Words Removal
  
stop_words = set(stopwords.words('english'))

n_documents = len(d["Features"])
for i in range(n_documents):
    d["Features"][i] = [w for w in d["Features"][i] if not w in stop_words]
    s = ""
    for el in d["Features"][i]:
        s += el + " "
    d["Features"][i] = s[:-1]
     
df = pd.DataFrame(data=d)

# # # #
# 3. Construct tf_idf matrix
# # # #

count_vectorizer = CountVectorizer()
counts_matrix = count_vectorizer.fit_transform(d["Features"])
doc_term_matrix = counts_matrix.todense()
euclidean_matrix = euclidean_distances(counts_matrix)

# # # #
# 4. Create training and test sets
# # # #

trains, tests = train_test_split(df, test_size=0.1, random_state=25)

# # # #
# 5. Classify testing data
# # # #

y_pred = []
y_true = []

for test_index, test_row in tests.iterrows():
    euclidean = {'=Poor=': [0,0], '=Unsatisfactory=': [0,0], '=Good=': [0,0], '=VeryGood=': [0,0], '=Excellent=': [0,0]}
    for train_index, train_row in trains.iterrows():
        euclidean[train_row["Labels"]][0] += euclidean_matrix[train_index][test_index]
        euclidean[train_row["Labels"]][1] += 1
    euclidean["=Poor="] = euclidean["=Poor="][0] / euclidean["=Poor="][1]
    euclidean["=Unsatisfactory="] = euclidean["=Unsatisfactory="][0] / euclidean["=Unsatisfactory="][1]
    euclidean["=Good="] = euclidean["=Good="][0] / euclidean["=Good="][1]
    euclidean["=VeryGood="] = euclidean["=VeryGood="][0] / euclidean["=VeryGood="][1]
    euclidean["=Excellent="] = euclidean["=Excellent="][0] / euclidean["=Excellent="][1]
    
    y_pred.append(min(euclidean, key=euclidean.get))
    y_true.append(test_row["Labels"])
    
print(accuracy_score(y_true, y_pred))