# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 5th model: naive bayes classifier (with some spice)
# # # # # # # # # # # # # # # # # # # # # # # # # # #

from copy import deepcopy
import sys
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def good_bad_adjectives(features, labels):
    tags = [nltk.pos_tag(feature) for feature in features]
    
    good, bad = [], []
    n_features = len(features)
    for i in range(n_features):
        for elm in tags[i]:
            if elm[1] in ('JJ', 'JJR', 'JJS'):
                if labels[i] in ('=Good=', '=VeryGood=', '=Excellent='):
                    good.append(elm[0])
                elif labels[i] in ('=Poor=', '=Unsatisfactory='):
                    bad.append(elm[0])
                else:
                    print("ERROR")
    return good, bad

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

features, train_features = [], []

with open(train) as f:
    lines = f.readlines()

for line in lines:
    feature = line.split()[1:]
    s = ""
    for f in feature:
        s += f + " "
    s = s[:-1]
    features.append(s)
    train_features.append(s)

with open(test) as f:
    lines = f.readlines()

for line in lines:
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
# 4. Pre-process reviews
# # # #

# 4.1 Lowercasing
features = [feature.lower() for feature in features]
train_features = [feature.lower() for feature in train_features]

# 4.2 Tokenization
features = [nltk.word_tokenize(feature) for feature in features]
train_features = [nltk.word_tokenize(feature) for feature in train_features]

# 4.3 Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
features = [ [wordnet_lemmatizer.lemmatize(word) for word in feature] for feature in features]
features_before = deepcopy(features)

train_features = [ [wordnet_lemmatizer.lemmatize(word) for word in feature] for feature in train_features]

# 4.4 Stop Words Removal

stop_words = set(stopwords.words('english'))

n_documents = len(features)
for i in range(n_documents):
    features[i] = [w for w in features[i] if not w in stop_words]
    s = ""
    for el in features[i]:
        s += el + " "
    features[i] = s[:-1]

n_t_documents = len(train_features)
for i in range(n_t_documents):
    train_features[i] = [w for w in train_features[i] if not w in stop_words]

# # # #
# 5. Prepare features for naive bayes model
# # # #

count_vectorizer = CountVectorizer()
counts_matrix    = count_vectorizer.fit_transform(features)
doc_term_matrix  = counts_matrix.todense()

df = pd.DataFrame(doc_term_matrix)

# # # #
# 6. Add more features
# # # #

# 6.1 Number of adjectives
tags = [nltk.pos_tag(feature) for feature in features_before]
list_of_adjectives = []
for i in range(n_documents):
    n_adjectives = 0
    for elm in tags[i]:
        if elm[1] in ('JJ', 'JJR', 'JJS'):
            n_adjectives += 1
    list_of_adjectives.append(n_adjectives)

df['N_adj'] = list_of_adjectives

# 5.2 Number of "positive" adjectives
'''
good, bad = good_bad_adjectives(train_features, train_labels)
list_of_good = []
for i in range(n_documents):
    n_good = 0
    for elm in tags[i]:
        if elm[0] in good:
            n_good += 1
    list_of_good.append(n_good)

df['N_good'] = list_of_good
print(list_of_good)
'''   

# 5.3 Number of "negative" adjectives

# 5.4 Presence of negative words (no, not, n't)
tags = [nltk.pos_tag(feature) for feature in features_before]
list_of_negatives = []
for i in range(n_documents):
    negative = 0
    for elm in tags[i]:
        if elm[0] in ('no', 'not', 'n\'t'):
            negative = 1
    list_of_negatives.append(negative)

df['Neg?'] = list_of_negatives

# 5.5 Number of "!"
'''
tags = [nltk.pos_tag(feature) for feature in features_before]
list_of_exclamations = []
for i in range(n_documents):
    n_exclamations = 0
    for elm in tags[i]:
        if elm[1] in ('.') and elm[0] in ("!"):
            n_exclamations += 1
    list_of_exclamations.append(n_exclamations)

df['N_exl'] = list_of_exclamations
'''

# 5.6 Number of "?"
'''
tags = [nltk.pos_tag(feature) for feature in features_before]
list_of_interrogations = []
for i in range(n_documents):
    n_interrogations = 0
    for elm in tags[i]:
        if elm[1] in ('.') and elm[0] in ("?"):
            n_interrogations += 1
    list_of_interrogations.append(n_interrogations)

df['N_int'] = list_of_interrogations
'''

# 5.7 Number of "..."
tags = [nltk.pos_tag(feature) for feature in features_before]
list_of_rets = []
for i in range(n_documents):
    n_rets = 0
    for elm in tags[i]:
        if elm[1] in ('.') and elm[0] in ("..."):
            n_rets += 1
    list_of_rets.append(n_rets)

df['N_rets'] = list_of_rets

# 5.8 Number of ".."
'''
tags = [nltk.pos_tag(feature) for feature in features_before]
list_of_doub = []
for i in range(n_documents):
    n_doub = 0
    for elm in tags[i]:
        if elm[1] in ('.') and elm[0] in (".."):
            n_doub += 1
    list_of_doub.append(n_doub)

df['N_doub'] = list_of_doub
'''

# 5.9 Number of ":("
# 5.10 Number of ":)"
# 5.11 Number of ":-("

# # # #
# 7. Learn model
# # # #

model = MultinomialNB().fit(df.loc[0:8999], train_labels)

# # # #
# 8. Predict test labels
# # # #
y_pred = model.predict(df.loc[9000:9999])

print('Accuracy:', accuracy_score(test_labels, y_pred))