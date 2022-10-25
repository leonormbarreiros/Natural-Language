# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Exploratory analysis of the dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
import matplotlib.pyplot as plt

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
# 2. Distribution of the training data
# # # #

# creating the dataset
data = {'=Poor=': d["Labels"].count('=Poor='), 
        '=Unsatisfactory=': d["Labels"].count('=Unsatisfactory='),
        '=Good=': d["Labels"].count('=Good='),
        '=VeryGood=': d["Labels"].count('=VeryGood='),
        '=Excellent=': d["Labels"].count('=Excellent=')
       }
labels = list(data.keys())
counts = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(labels, counts, color ='maroon',
        width = 0.4)
 
plt.xlabel("Classifications")
plt.ylabel("No. of labelled entries")
plt.title("Classifications of Beauty Products")
#plt.show()

# # # #
# 3. Length of each review
# # # #

df['len'] = df['Features'].str.len()
df = df.sort_values(['len'], ascending=True)
#print(df.head(50))