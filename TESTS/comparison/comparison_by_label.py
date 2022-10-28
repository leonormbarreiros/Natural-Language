X = ["run 1", "run 2", "run 3", "run 4", "run 5", "run 6"]

# LR
lr_totl = [0.4955, 0.5000, 0.4870, 0.4720, 0.4860, 0.4860]
lr_poor = [0.6556, 0.5681, 0.5547, 0.5179, 0.5189, 0.5254]
lr_unsa = [0.3916, 0.4192, 0.3787, 0.3760, 0.4390, 0.4016]
lr_good = [0.5475, 0.4061, 0.3951, 0.3626, 0.4015, 0.3981]
lr_very = [0.4225, 0.4655, 0.4338, 0.4770, 0.4090, 0.4415]
lr_exce = [0.4723, 0.6466, 0.6566, 0.6296, 0.6667, 0.6728]

# NB1
nb1_totl = [0.4915, 0.5005, 0.4815, 0.4635, 0.4910, 0.4815]
nb1_poor = [0.6122, 0.5964, 0.5729, 0.5282, 0.5519, 0.5593]
nb1_unsa = [0.3916, 0.3911, 0.3746, 0.3581, 0.4220, 0.4094]
nb1_good = [0.5625, 0.4086, 0.3902, 0.3649, 0.3966, 0.3621]
nb1_very = [0.4225, 0.5090, 0.4559, 0.4668, 0.4433, 0.4610]
nb1_exce = [0.4776, 0.6065, 0.6195, 0.6025, 0.6457, 0.6227]

# NB2
nb2_totl = [0.4915, 0.5005, 0.4820, 0.4635, 0.4905, 0.4810]
nb2_poor = [0.6122, 0.5964, 0.5729, 0.5282, 0.5519, 0.5569]
nb2_unsa = [0.3916, 0.3911, 0.3746, 0.3581, 0.4220, 0.4094]
nb2_good = [0.5625, 0.4086, 0.3902, 0.3649, 0.3941, 0.3621]
nb2_very = [0.4225, 0.5090, 0.4583, 0.4668, 0.4433, 0.4610]
nb2_exce = [0.4776, 0.6065, 0.6195, 0.6025, 0.6457, 0.6227]

# SVM1
svm1_totl = [0.4900, 0.4935, 0.4785, 0.4690, 0.4785, 0.4810]
svm1_poor = [0.7092, 0.6504, 0.6224, 0.5641, 0.5802, 0.6029]
svm1_unsa = [0.3427, 0.3326, 0.3079, 0.3325, 0.3780, 0.3438]
svm1_good = [0.6075, 0.3477, 0.3220, 0.2986, 0.3177, 0.3237]
svm1_very = [0.3425, 0.4629, 0.4191, 0.4439, 0.4063, 0.4293]
svm1_exce = [0.4617, 0.6867, 0.7007, 0.7111, 0.7165, 0.7150]

# SVM2
svm2_totl = [0.4900, 0.4950, 0.4800, 0.4680, 0.4785, 0.4795]
svm2_poor = [0.7041, 0.6452, 0.6250, 0.5615, 0.5778, 0.6053]
svm2_unsa = [0.3473, 0.3396, 0.3052, 0.3299, 0.3829, 0.3386]
svm2_good = [0.6100, 0.3528, 0.3220, 0.2986, 0.3153, 0.3213]
svm2_very = [0.3400, 0.4629, 0.4191, 0.4439, 0.4063, 0.4293]
svm2_exce = [0.4617, 0.6867, 0.7077, 0.7111, 0.7165, 0.7124]

# baseline 1
base1_totl = [0.3115, 0.3215, 0.3215, 0.3090, 0.3115, 0.3300]
base1_poor = [0.2474, 0.2381, 0.2854, 0.3309, 0.2677, 0.3087]
base1_unsa = [0.1911, 0.1497, 0.2341, 0.1564, 0.1650, 0.1727]
base1_good = [0.0450, 0.0514, 0.0468, 0.0462, 0.0401, 0.0460]
base1_very = [0.5000, 0.4848, 0.4496, 0.4961, 0.5488, 0.5013]
base1_exce = [0.5963, 0.6701, 0.5907, 0.5255, 0.5594, 0.6366]

# baseline 2
base2_totl = [0.4780, 0.4815, 0.4710, 0.4590, 0.4600, 0.4630]
base2_poor = [0.6658, 0.6642, 0.6798, 0.6839, 0.6877, 0.6939]
base2_unsa = [0.3499, 0.3528, 0.3317, 0.2986, 0.3054, 0.3261]
base2_good = [0.5900, 0.6067, 0.5964, 0.5487, 0.5401, 0.5424]
base2_very = [0.3450, 0.3396, 0.3079, 0.3376, 0.3805, 0.3465]
base2_exce = [0.4565, 0.4552, 0.4191, 0.4311, 0.3931, 0.4171]

import matplotlib.pyplot as plt

plt.subplot(2, 3, 1)
plt.plot(X, lr_totl, label='Logistic Regression')
plt.plot(X, nb1_totl, label='Naive Bayes')
#plt.plot(X, nb2_totl, label='Naive Bayes 2')
plt.plot(X, svm1_totl, label='Suppor Vector Machine')
#plt.plot(X, svm2_totl, label='Suppor Vector Machine 2')
plt.plot(X, base1_totl, label='Baseline 1', linestyle='-')
plt.plot(X, base2_totl, label='Baseline 2', linestyle='-')
plt.title("Overall")

plt.subplot(2, 3, 2)
plt.plot(X, lr_poor, label='Logistic Regression')
plt.plot(X, nb1_poor, label='Naive Bayes')
#plt.plot(X, nb2_poor, label='Naive Bayes 2')
plt.plot(X, svm1_poor, label='Suppor Vector Machine')
#plt.plot(X, svm2_poor, label='Suppor Vector Machine 2')
plt.plot(X, base1_poor, label='Baseline 1', linestyle='-')
plt.plot(X, base2_poor, label='Baseline 2', linestyle='-')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),ncol=3, fancybox=True, shadow=True)
plt.title("Poor")

plt.subplot(2, 3, 3)
plt.plot(X, lr_unsa, label='Logistic Regression')
plt.plot(X, nb1_unsa, label='Naive Baye')
#plt.plot(X, nb2_unsa, label='Naive Bayes 2')
plt.plot(X, svm1_unsa, label='Suppor Vector Machine')
#plt.plot(X, svm2_unsa, label='Suppor Vector Machine 2')
plt.plot(X, base1_unsa, label='Baseline 1', linestyle='-')
plt.plot(X, base2_unsa, label='Baseline 2', linestyle='-')
plt.title("Unsatisfactory")

plt.subplot(2, 3, 4)
plt.plot(X, lr_good, label='Logistic Regression')
plt.plot(X, nb1_good, label='Naive Bayes')
#plt.plot(X, nb2_good, label='Naive Bayes 2')
plt.plot(X, svm1_good, label='Suppor Vector Machine')
#plt.plot(X, svm2_good, label='Suppor Vector Machine 2')
plt.plot(X, base1_good, label='Baseline 1', linestyle='-')
plt.plot(X, base2_good, label='Baseline 2', linestyle='-')
plt.title("Good")

plt.subplot(2, 3, 5)
plt.plot(X, lr_very, label='Logistic Regression')
plt.plot(X, nb1_very, label='Naive Bayes')
#plt.plot(X, nb2_very, label='Naive Bayes 2')
plt.plot(X, svm1_very, label='Suppor Vector Machine')
#plt.plot(X, svm2_very, label='Suppor Vector Machine 2')
plt.plot(X, base1_very, label='Baseline 1', linestyle='-')
plt.plot(X, base2_very, label='Baseline 2', linestyle='-')
plt.title("VeryGood")

plt.subplot(2, 3, 6)
plt.plot(X, lr_exce, label='Logistic Regression')
plt.plot(X, nb1_exce, label='Naive Bayes')
#plt.plot(X, nb2_exce, label='Naive Bayes 2')
plt.plot(X, svm1_exce, label='Suppor Vector Machine')
#plt.plot(X, svm2_exce, label='Suppor Vector Machine 2')
plt.plot(X, base1_exce, label='Baseline 1', linestyle='-')
plt.plot(X, base2_exce, label='Baseline 2', linestyle='-')
plt.title("Excellent")

plt.show()