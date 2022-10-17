# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2nd model: classify according to Dice
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def DICE(s, t):
    s_set = set(s)
    t_set = set(t)
    
    s_e_t = len(s_set.intersection(t_set))
    s_len = len(s_set)
    t_len = len(t_set)
    
    return (2 * s_e_t) / (s_len + t_len)

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
    dice = {'=Poor=': [0,0], '=Unsatisfactory=': [0,0], '=Good=': [0,0], '=VeryGood=': [0,0], '=Excellent=': [0,0]}
    for train_index, train_row in trains.iterrows():
        dice[train_row["Labels"]][0] += DICE(train_row["Features"], test_row["Features"])
        dice[train_row["Labels"]][1] += 1
    dice["=Poor="] = dice["=Poor="][0] / dice["=Poor="][1]
    dice["=Unsatisfactory="] = dice["=Unsatisfactory="][0] / dice["=Unsatisfactory="][1]
    dice["=Good="] = dice["=Good="][0] / dice["=Good="][1]
    dice["=VeryGood="] = dice["=VeryGood="][0] / dice["=VeryGood="][1]
    dice["=Excellent="] = dice["=Excellent="][0] / dice["=Excellent="][1]
    
    y_pred.append(max(dice, key=dice.get))
    y_true.append(test_row["Labels"])
    
print(accuracy_score(y_true, y_pred))