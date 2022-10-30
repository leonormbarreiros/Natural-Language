# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Sentiment Analysis of Beauty Products' Reviews
# 95617 Juliana Yang
# 95618 Leonor Barreiros
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import sys

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 0. Read arguments

args = sys.argv[1:]

i = 0
for arg in args:
    if arg == '-test':
        test = args[i + 1]
    if arg == '-train':
        train = args[i + 1]
    i += 1

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1. Create dataset

d = { "Features" : [], "Length": [] }

# # # # # # # # # # # 
# 1.1 Train dataset
train_labels = { "Labels" : [] }

with open(train) as f:
    lines = f.readlines()

for line in lines:
    label = line.split()[0]
    train_labels["Labels"].append(label)
    features = line.split()[1:]
    s = ""
    for feature in features:
        s += feature + " "
    s = s[:-1]
    
    d["Features"].append(s.lower()) # lowercasing

# # # # # # # # # # # 
# 1.2 Test dataset
with open(test) as f:
    lines_features = f.readlines()

for i in range(2000):
    feature = lines_features[i][:-1]
    d["Features"].append(feature.lower()) # lowercasing
    
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2. Pre-processing

d["Features"] = [nltk.word_tokenize(feature) for feature in d["Features"]]
wordnet_lemmatizer = WordNetLemmatizer()
d["Features"] = [ [wordnet_lemmatizer.lemmatize(word) for word in feature] for feature in d["Features"]]

for i in range(10000):
    s = ""
    for el in d["Features"][i]:
        s += el + " "
    d["Features"][i] = s[:-1]
   
count_vectorizer = CountVectorizer()
counts_matrix    = count_vectorizer.fit_transform(d["Features"])
doc_term_matrix  = counts_matrix.todense()

df = pd.DataFrame(doc_term_matrix)

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 3. Feature engineering

# 3.1 Number of adjectives
d["Features"] = [nltk.word_tokenize(feature) for feature in d["Features"]]
tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_adjectives = []
for i in range(10000):
    n_adjectives = 0
    for elm in tags[i]:
        if elm[1] in ('JJ', 'JJR', 'JJS'):
            n_adjectives += 1
    list_of_adjectives.append(n_adjectives)

df['N_adj'] = list_of_adjectives

# 3.2 Negative words
tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_negatives = []
for i in range(10000):
    negative = 0
    for elm in tags[i]:
        if elm[0] in ('no', 'not', 'n\'t'):
            negative = 1
    list_of_negatives.append(negative)

df['Neg?'] = list_of_negatives

# 3.3 Exclamation marks
tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_exclamations = []
for i in range(10000):
    n_exclamations = 0
    for elm in tags[i]:
        if elm[1] in ('.') and elm[0] in ("!"):
            n_exclamations += 1
    list_of_exclamations.append(n_exclamations)

df['N_exl'] = list_of_exclamations

# 3.4 :)
tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_smile = []
for i in range(10000):
    n_smile = 0
    for j in range( len(tags[i]) - 1 ):
        elm = tags[i][j]
        elm_next = tags[i][j + 1]
        #porque o segundo elemneto não é "."
        if elm == (':', ':') and elm_next == (')', ')'):
            #print("OK")
            n_smile += 1
    list_of_smile.append(n_smile)

df['N_smile'] = list_of_smile

# 3.5 :(
tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_sad = []
for i in range(10000):
    n_sad = 0
    for j in range( len(tags[i]) - 1 ):
        elm = tags[i][j]
        elm_next = tags[i][j + 1]
        #porque o segundo elemneto não é "."
        if elm == (':', ':') and elm_next == ('(', '('):
            n_sad += 1
    list_of_sad.append(n_sad)

df['N_sad'] = list_of_sad

# 3.6 Number of positive words and Number of negative words
list_of_positives = []
list_of_negatives = []
list_of_neutrals = []
for s in d["Features"]:
    n_positive = 0
    n_negative = 0
    n_neutral = 0
    for w in s:
        sentiment_dict = SentimentIntensityAnalyzer().polarity_scores(w)
        if sentiment_dict['compound'] >= 0.05:
            n_positive += 1
        elif sentiment_dict['compound'] <= - 0.05:
            n_negative += 1
        else:
            n_neutral += 1
    list_of_positives.append(n_positive)
    list_of_negatives.append(n_negative)
    list_of_neutrals.append(n_neutral)

df['N_positives'] = list_of_positives
df['N_negatives'] = list_of_negatives
df['N_neutrals'] = list_of_neutrals 


# 3.7 Positive sentence or Negative sentence?
sentences = []
list_of_positive = []
list_of_negative = []

for s in d["Features"]:
    #sentences.append(' '.join(s))
    list_of_negative.append(0)
    list_of_positive.append(0)
    sentiment_dict = SentimentIntensityAnalyzer().polarity_scores(' '.join(s))
    if sentiment_dict['compound'] >= 0.05:
        list_of_positive[-1] = 1
    elif sentiment_dict['compound'] <= - 0.05:
        list_of_negative[-1] = 1
df['N_positive'] = list_of_positive
df['N_negative'] = list_of_negative

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 4. Create classifier
pipeline = Pipeline(
    [
        ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
    ]
)

learner = pipeline.fit(df.loc[0:7999], train_labels['Labels'])

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 5. Predict and print test labels
y_pred = learner.predict(df.loc[8000:9999])

for pred in y_pred:
    print(pred)
