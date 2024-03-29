# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Classification using Logistic Regression
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import nltk

d = { "Features" : [], "Length": [] }

# # # #
# 1. Create train dataset by reading train_aux.txt
# # # #
train_labels = { "Labels" : [] }

with open('train_aux.txt') as f:
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

# # # #
# 1. Create test dataset by reading test.txt and true.txt
# # # #
test_labels = { "Labels" : [] }

with open('test.txt') as f:
    lines_features = f.readlines()
with open('true.txt') as f:
    lines_labels = f.readlines()

for i in range(2000):
    label = lines_labels[i][:-1]
    feature = lines_features[i][:-1]
    test_labels["Labels"].append(label)
    d["Features"].append(feature.lower()) # lowercasing

'''
# # # #
# 1.1 Pre-process the dataset
# # # #
d["Features"] = [nltk.word_tokenize(feature) for feature in d["Features"]]
wordnet_lemmatizer = WordNetLemmatizer()
d["Features"] = [ [wordnet_lemmatizer.lemmatize(word) for word in feature] for feature in d["Features"]]

for i in range(10000):
    s = ""
    for el in d["Features"][i]:
        s += el + " "
    d["Features"][i] = s[:-1]
'''    
count_vectorizer = CountVectorizer()
#count_vectorizer = TfidfVectorizer()
counts_matrix    = count_vectorizer.fit_transform(d["Features"])
doc_term_matrix  = counts_matrix.todense()

df = pd.DataFrame(doc_term_matrix)
#df["Length"] = [len(feature) for feature in d["Features"]]

# # # #
# 1.1 Extra features
# # # #
# 1.1.1 Number of adjectives
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

# 1.1.2 Negative words
#d["Features"] = [nltk.word_tokenize(feature) for feature in d["Features"]]
tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_negatives = []
for i in range(10000):
    negative = 0
    for elm in tags[i]:
        if elm[0] in ('no', 'not', 'n\'t'):
            negative = 1
    list_of_negatives.append(negative)

df['Neg?'] = list_of_negatives

# 1.1.3 Exclamation marks
#d["Features"] = [nltk.word_tokenize(feature) for feature in d["Features"]]

tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_exclamations = []
for i in range(10000):
    n_exclamations = 0
    for elm in tags[i]:
        if elm[1] in ('.') and elm[0] in ("!"):
            n_exclamations += 1
    list_of_exclamations.append(n_exclamations)

df['N_exl'] = list_of_exclamations

# 1.1.4 "..."
""" tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_rets = []
for i in range(10000):
    n_rets = 0
    for elm in tags[i]:
        if elm[1] in ('.') and elm[0] in ("..."):
            n_rets += 1
    list_of_rets.append(n_rets)

df['N_rets'] = list_of_rets """

# 1.1.5 "?"
tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_interrogations = []
for i in range(10000):
    n_interrogations = 0
    for elm in tags[i]:
        if elm[1] in ('.') and elm[0] in ("?"):
            n_interrogations += 1
    list_of_interrogations.append(n_interrogations)

df['N_int'] = list_of_interrogations

# 1.1.6 :-(
tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_big_sad = []
for i in range(10000):
    n_big_sad = 0
    for j in range( len(tags[i]) - 2 ):
        elm = tags[i][j]
        elm_next = tags[i][j + 1]
        elm_next_next = tags[i][j + 2]
        #porque o segundo elemneto não é "."
        if elm == (':', ':') and elm_next == ('-', ':') and elm_next_next == ('(', '('):
            n_big_sad += 1
    list_of_big_sad.append(n_big_sad)

df['N_big_sad'] = list_of_big_sad   

# 1.1.7 :)
""" tags = [nltk.pos_tag(feature) for feature in d["Features"]]
list_of_smile = []
for i in range(10000):
    n_smile = 0
    for j in range( len(tags[i]) - 1 ):
        elm = tags[i][j]
        elm_next = tags[i][j + 1]
        #porque o segundo elemneto não é "."
        if elm == (':', ':') and elm_next == (')', ')'):
            n_smile += 1
    list_of_smile.append(n_smile)

df['N_smile'] = list_of_smile """

# 1.1.8 :(
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


# 1.1.9 Number of positive words and Number of negative words
""" list_of_positives = []
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
df['N_neutrals'] = list_of_neutrals  """


# 1.1.9 Positive sentence or Negative sentence?
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


# # # #
# 2. Initialize classifier
# # # #
pipeline = Pipeline(
    [
        ('clf', SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=1e-3,
            random_state=42,
            max_iter=100,
            learning_rate='optimal',
            tol=None
        )),
    ]
)

#print(df.loc[8000:9999])

learner = pipeline.fit(df.loc[0:7999], train_labels['Labels'])

# # # #
# 3. Predict test labels
# # # #
y_pred = learner.predict(df.loc[8000:9999])

# # # #
# 4. Evaluate performance
# # # #

# 4.1 Confusion matrix
# i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
import matplotlib.pyplot as plt
from sklearn import metrics
actual = test_labels['Labels']
predicted = y_pred
labels=['Poor', 'Unsatisfactory', 'Good', 'VeryGood', 'Excellent']

confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=['=Poor=', '=Unsatisfactory=', '=Good=', '=VeryGood=', '=Excellent='])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)

cm_display.plot()
plt.show()

# 4.2 Accuracy
print('Accuracy:', accuracy_score(test_labels['Labels'], y_pred))

# 4.3 Accuracy by Label
poor_acc = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[0, 2] + confusion_matrix[0, 3] + confusion_matrix[0, 4])
unsa_acc = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[1, 3] + confusion_matrix[1, 4])
good_acc = confusion_matrix[2, 2] / (confusion_matrix[2, 0] + confusion_matrix[2, 1] + confusion_matrix[2, 2] + confusion_matrix[2, 3] + confusion_matrix[2, 4])
very_acc = confusion_matrix[3, 3] / (confusion_matrix[3, 0] + confusion_matrix[3, 1] + confusion_matrix[3, 2] + confusion_matrix[3, 3] + confusion_matrix[3, 4])
exce_acc = confusion_matrix[4, 4] / (confusion_matrix[4, 0] + confusion_matrix[4, 1] + confusion_matrix[4, 2] + confusion_matrix[4, 3] + confusion_matrix[4, 4])

print('Poor Accuracy: ', poor_acc)
print('Unsatisfactory Accuracy: ', unsa_acc)
print('Good Accuracy: ', good_acc)
print('VeryGood Accuracy: ', very_acc)
print('Excellent Accuracy: ', exce_acc)

# 4.4 Save errors to file (by label)
f_poor = open('poor.txt', 'w')
f_unsa = open('unsatisfactory.txt', 'w')
f_good = open('good.txt', 'w')
f_very = open('verygood.txt', 'w')
f_exce = open('excellent.txt', 'w')

for i in range(10000):
    s = ""
    for el in d["Features"][i]:
        s += el + " "
    d["Features"][i] = s[:-1]
    
for i in range(2000):
    true_label = actual[i]
    pred_label = predicted[i]
    
    if true_label != pred_label:
        if true_label == '=Poor=':
            f_poor.write(pred_label + '                 ' + d['Features'][i + 8000])
            f_poor.write('\n')
        elif true_label == '=Unsatisfactory=':
            f_unsa.write(pred_label + '                 ' + d['Features'][i + 8000])
            f_unsa.write('\n')
        elif true_label == '=Good=':
            f_good.write(pred_label + '                 ' + d['Features'][i + 8000])
            f_good.write('\n')
        elif true_label == '=VeryGood=':    
            f_very.write(pred_label + '                 ' + d['Features'][i + 8000])
            f_very.write('\n')
        elif true_label == '=Excellent=':
            f_exce.write(pred_label + '                 ' + d['Features'][i + 8000])
            f_exce.write('\n')
        else:
            print('ERROR')

'''
Accuracy: 0.49
Poor Accuracy:  0.7091836734693877
Unsatisfactory Accuracy:  0.34265734265734266
Good Accuracy:  0.6075
VeryGood Accuracy:  0.3425
Excellent Accuracy:  0.46174142480211083
'''