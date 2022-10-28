# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2nd model: classify according to Dice
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def JACCARD(s, t):
    s_set = set(s)
    t_set = set(t)
    
    s_e_t = len(s_set.intersection(t_set))
    s_u_t = len(s_set.union(t_set))
    
    return s_e_t / s_u_t

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

# # # #
# 4. Classify testing data
# # # #

y_pred = []
y_true = []

for i in range(8000, 10000):
    jaccard = {'=Poor=': [0,0], '=Unsatisfactory=': [0,0], '=Good=': [0,0], '=VeryGood=': [0,0], '=Excellent=': [0,0]}
    for j in range(0, 8000):
        jaccard[train_labels["Labels"][j]][0] += JACCARD(d["Features"][j], d["Features"][i])
        jaccard[train_labels["Labels"][j]][1] += 1
    jaccard["=Poor="] = jaccard["=Poor="][0] / jaccard["=Poor="][1]
    jaccard["=Unsatisfactory="] = jaccard["=Unsatisfactory="][0] / jaccard["=Unsatisfactory="][1]
    jaccard["=Good="] = jaccard["=Good="][0] / jaccard["=Good="][1]
    jaccard["=VeryGood="] = jaccard["=VeryGood="][0] / jaccard["=VeryGood="][1]
    jaccard["=Excellent="] = jaccard["=Excellent="][0] / jaccard["=Excellent="][1]
    
    y_pred.append(max(jaccard, key=jaccard.get))
    y_true.append(test_labels["Labels"][i - 8000])
    
# # # #
# 4. Evaluate performance
# # # #

# 4.1 Confusion matrix
# i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
import matplotlib.pyplot as plt
from sklearn import metrics
actual = y_true
predicted = y_pred
labels=['Poor', 'Unsatisfactory', 'Good', 'VeryGood', 'Excellent']

confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)

cm_display.plot()
plt.show()

# 4.2 Accuracy
print('Accuracy:', accuracy_score(y_true, y_pred))

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

Accuracy: 0.3115
Poor Accuracy:  0.24744897959183673
Unsatisfactory Accuracy:  0.19114219114219114
Good Accuracy:  0.045
VeryGood Accuracy:  0.5
Excellent Accuracy:  0.5963060686015831
'''

'''
727

Accuracy: 0.3215
Poor Accuracy:  0.23809523809523808
Unsatisfactory Accuracy:  0.14974619289340102
Good Accuracy:  0.05141388174807198
VeryGood Accuracy:  0.4847775175644028
Excellent Accuracy:  0.670076726342711
'''

'''
836

Accuracy: 0.3215
Poor Accuracy:  0.2853828306264501
Unsatisfactory Accuracy:  0.23414634146341465
Good Accuracy:  0.046875
VeryGood Accuracy:  0.44959128065395093
Excellent Accuracy:  0.5906862745098039
'''

'''
67

Accuracy: 0.309
Poor Accuracy:  0.3308641975308642
Unsatisfactory Accuracy:  0.15639810426540285
Good Accuracy:  0.046153846153846156
VeryGood Accuracy:  0.4961636828644501
Excellent Accuracy:  0.5255102040816326
'''

'''
47

Accuracy: 0.3115
Poor Accuracy:  0.2677165354330709
Unsatisfactory Accuracy:  0.16502463054187191
Good Accuracy:  0.04009433962264151
VeryGood Accuracy:  0.5487804878048781
Excellent Accuracy:  0.5593667546174143
'''

'''
215

Accuracy: 0.33
Poor Accuracy:  0.3087071240105541
Unsatisfactory Accuracy:  0.17266187050359713
Good Accuracy:  0.04600484261501211
VeryGood Accuracy:  0.5013123359580053
Excellent Accuracy:  0.6365853658536585
'''