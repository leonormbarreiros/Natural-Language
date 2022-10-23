# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2nd model: classify according to cosine similarity
# # # # # # # # # # # # # # # # # # # # # # # # # # #

from copy import deepcopy
import sys
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# # # #
# 1. Get train and test sets' files
# # # #

args = sys.argv[1:]

i = 0
for arg in args:
    if arg == '-test':
        test = args[i + 1]
    if arg == '-train':
        train = args[i + 1]
    i += 1

# # # #
# 2. Get train & test features
# # # #

features = []

with open(train) as f:
    lines = f.readlines()

for line in lines:
    feature = line.split()[1:]
    s = ""
    for f in feature:
        s += f + " "
    s = s[:-1]
    features.append(s)

with open(test) as f:
    lines = f.readlines()

for line in lines:
    features.append(line)

# # # #
# 3. Get train & test labels
# # # #

labels = []

with open(train) as f:
    lines = f.readlines()

for line in lines:
    label = line.split()[0]
    labels.append(label)

test_labels = []

with open('test_results.txt') as f:
    lines = f.readlines()

for line in lines:
    test_labels.append(line[:-1])

# # # #
# 4. Process reviews
# # # #

# 4.1 Lowercasing
features = [feature.lower() for feature in features]

# 4.2 Tokenization
features = [nltk.word_tokenize(feature) for feature in features]

# 4.3 Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
features = [ [wordnet_lemmatizer.lemmatize(word) for word in feature] for feature in features]
features_before = deepcopy(features)

# 4.4 Stop Words Removal

stop_words = set(stopwords.words('english'))

n_documents = len(features)
for i in range(n_documents):
    features[i] = [w for w in features[i] if not w in stop_words]
    s = ""
    for el in features[i]:
        s += el + " "
    features[i] = s[:-1]

# # # #
# 5. Construct tf_idf matrix
# # # #

count_vectorizer = CountVectorizer()
counts_matrix = count_vectorizer.fit_transform(features)
doc_term_matrix = counts_matrix.todense()
cosine_similarity_matrix = cosine_similarity(counts_matrix)

df = pd.DataFrame(doc_term_matrix)

# # # #
# 6. Classify testing data
# # # #

y_pred = []
y_true = []

for i in range(8000, n_documents):
    cosine = {'=Poor=': [0,0], '=Unsatisfactory=': [0,0], '=Good=': [0,0], '=VeryGood=': [0,0], '=Excellent=': [0,0]}
    for j in range(0, 8000):
        cosine[labels[j]][0] += cosine_similarity_matrix[i][j]
        cosine[labels[j]][1] += 1
    
    cosine["=Poor="] = cosine["=Poor="][0] / cosine["=Poor="][1]
    cosine["=Unsatisfactory="] = cosine["=Unsatisfactory="][0] / cosine["=Unsatisfactory="][1]
    cosine["=Good="] = cosine["=Good="][0] / cosine["=Good="][1]
    cosine["=VeryGood="] = cosine["=VeryGood="][0] / cosine["=VeryGood="][1]
    cosine["=Excellent="] = cosine["=Excellent="][0] / cosine["=Excellent="][1]
    
    pred = max(cosine, key=cosine.get)
    y_pred.append(pred)
    print(pred)

print(accuracy_score(test_labels, y_pred))
    