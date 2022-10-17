# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1st model: classify according to MED
# # # # # # # # # # # # # # # # # # # # # # # # # # #
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def MED(s, t):
    n = len(s) - 1
    m = len(t) - 1
    
    C1, C2, C3 = 1, 1, 1
    
    if n == 0:
        return m
    if m == 0:
        return n
    
    M = [[0 for j in range(0, n + 1)] for i in range(0, m + 1)]
    for i in range(0, m + 1):
        M[i][0] = i
    for j in range(0, n + 1):
        M[0][j] = j
    
    j = 1
    while j < n + 1:
        i = 1
        while i < m + 1:
            if s[j] == t[i]:
                M[i][j] = M[i - 1][j - 1]
            else:
                M[i][j] = min(M[i - 1][j] + C1, M[i][j - 1] + C2, M[i - 1][j - 1] + C3)
            i += 1
        j += 1
    
    return M[m][n]

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

# 2.3 Stop Words Removal TODO

# 2.4 Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
d["Features"] = [ [wordnet_lemmatizer.lemmatize(word) for word in feature] for feature in d["Features"]]

df = pd.DataFrame(data=d)

# # # #
# 3. Create training and test sets
# # # #

trains, tests = train_test_split(df, test_size=0.1, random_state=25)

# # # #
# 4. Classify testing data
# # # #

y_pred = []
y_true = []

for test_index, test_row in tests.iterrows():
    med = {'=Poor=': [0,0], '=Unsatisfactory=': [0,0], '=Good=': [0,0], '=VeryGood=': [0,0], '=Excellent=': [0,0]}
    for train_index, train_row in trains.iterrows():
        med[train_row["Labels"]][0] += MED(train_row["Features"], test_row["Features"])
        med[train_row["Labels"]][1] += 1
    med["=Poor="] = med["=Poor="][0] / med["=Poor="][1]
    med["=Unsatisfactory="] = med["=Unsatisfactory="][0] / med["=Unsatisfactory="][1]
    med["=Good="] = med["=Good="][0] / med["=Good="][1]
    med["=VeryGood="] = med["=VeryGood="][0] / med["=VeryGood="][1]
    med["=Excellent="] = med["=Excellent="][0] / med["=Excellent="][1]
    
    y_pred.append(min(med, key=med.get))
    y_true.append(test_row["Labels"])
    
print(accuracy_score(y_true, y_pred))