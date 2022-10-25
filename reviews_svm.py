# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 4th model: naive bayes classifier
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

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

train_features, test_features, features = [], [], []

with open(train) as f:
    lines = f.readlines()

for line in lines:
    feature = line.split()[1:]
    s = ""
    for f in feature:
        s += f + " "
    s = s[:-1]
    train_features.append(s)
    features.append(s)

with open(test) as f:
    lines = f.readlines()

for line in lines:
    test_features.append(line)
    features.append(line)

# # # #
# 3. Get train & test labels
# # # #

train_labels = []

with open(train) as f:
    lines = f.readlines()

for line in lines:
    label = line.split()[0]
    train_labels.append(label)

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
# 5. Prepare features
# # # #

count_vectorizer = CountVectorizer()

counts_matrix   = count_vectorizer.fit_transform(features)
doc_term_matrix = counts_matrix.todense()

# # # #
# 6. Learn model
# # # #
model = svm.SVC(kernel='linear').fit(doc_term_matrix[:8000], train_labels)

# # # #
# 7. Predict test labels
# # # #
y_pred = model.predict(doc_term_matrix[8000:])

print('Accuracy:', accuracy_score(test_labels, y_pred))
