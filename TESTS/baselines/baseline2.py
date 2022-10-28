# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Classification using Logistic Regression
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
    
    d["Features"].append(s)

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
    d["Features"].append(feature)

count_vectorizer = CountVectorizer()
counts_matrix    = count_vectorizer.fit_transform(d["Features"])
doc_term_matrix  = counts_matrix.todense()

df = pd.DataFrame(doc_term_matrix)

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

confusion_matrix = metrics.confusion_matrix(actual, predicted)
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
           
'''
42

Accuracy: 0.478
Poor Accuracy:  0.6658163265306123
Unsatisfactory Accuracy:  0.34498834498834496
Good Accuracy:  0.59
VeryGood Accuracy:  0.345
Excellent Accuracy:  0.45646437994722955
'''

'''
727

Accuracy: 0.4815
Poor Accuracy:  0.6641604010025063
Unsatisfactory Accuracy:  0.35279187817258884
Good Accuracy:  0.6066838046272494
VeryGood Accuracy:  0.3395784543325527
Excellent Accuracy:  0.45524296675191817
'''

'''
836

Accuracy: 0.471
Poor Accuracy:  0.679814385150812
Unsatisfactory Accuracy:  0.33170731707317075
Good Accuracy:  0.5963541666666666
VeryGood Accuracy:  0.3079019073569482
Excellent Accuracy:  0.41911764705882354
'''

'''
67

Accuracy: 0.459
Poor Accuracy:  0.6839506172839506
Unsatisfactory Accuracy:  0.2985781990521327
Good Accuracy:  0.5487179487179488
VeryGood Accuracy:  0.3375959079283887
Excellent Accuracy:  0.43112244897959184
'''

'''
47

Accuracy: 0.46
Poor Accuracy:  0.6876640419947506
Unsatisfactory Accuracy:  0.3054187192118227
Good Accuracy:  0.5400943396226415
VeryGood Accuracy:  0.3804878048780488
Excellent Accuracy:  0.39313984168865435
'''

'''
215

Accuracy: 0.463
Poor Accuracy:  0.6939313984168866
Unsatisfactory Accuracy:  0.3261390887290168
Good Accuracy:  0.5423728813559322
VeryGood Accuracy:  0.3464566929133858
Excellent Accuracy:  0.4170731707317073
'''