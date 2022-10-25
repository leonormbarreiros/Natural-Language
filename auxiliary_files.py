# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Creates a dataframe and splits it into train & test
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
from sklearn.model_selection import train_test_split

# # # #
# 1. Create dataframe by reading train.txt
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

df = pd.DataFrame(data=d)

# # # #
# 2. Split into training and testing sets
# # # #
training_data, testing_data = train_test_split(df, test_size=0.2, random_state=21)

# # # #
# 3. Save training set
# # # #
f1 = open('train_aux.txt', 'w')
for train_index, train_row in training_data.iterrows():
    f1.write(lines[train_index])
    
# # # #
# 4. Save dev set (split between features and labels)
# # # #
f2 = open('test.txt', 'w')
f3 = open('true.txt', 'w')
for test_index, test_row in testing_data.iterrows():
    feature = test_row["Features"]
    f2.write(feature)
    f2.write('\n')
    label = test_row["Labels"]
    f3.write(label)
    f3.write('\n')