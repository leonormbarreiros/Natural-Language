# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Creates a dataframe and splits it into train & test
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
from sklearn.model_selection import train_test_split

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

training_data, testing_data = train_test_split(df, test_size=0.1, random_state=25)

# create tests file
for test_index, test_row in testing_data.iterrows():
    #feature = test_row["Features"]
    #print(feature)
    label = test_row["Labels"]
    print(label)

# create auxiliary trains file
#for train_index, train_row in training_data.iterrows():
#    print(lines[train_index][:-1])
