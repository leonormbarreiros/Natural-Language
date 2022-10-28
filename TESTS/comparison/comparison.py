import matplotlib.pyplot as plt

# data
X = ["run 1", "run 2", "run 3", "run 4", "run 5", "run 6"]

y_LR    = [0.4955, 0.5000, 0.4870, 0.4720, 0.4860, 0.4860]
y_NB1   = [0.4915, 0.5005, 0.4815, 0.4635, 0.4910, 0.4815]
#y_NB2   = [0.4915, 0.5005, 0.4820, 0.4635, 0.4905, 0.4810]
y_SVM1  = [0.4900, 0.4935, 0.4785, 0.4690, 0.4785, 0.4810]
#y_SVM2  = [0.4900, 0.4950, 0.4800, 0.4680, 0.4785, 0.4795]
y_base1 = [0.3115, 0.3215, 0.3215, 0.3090, 0.3115, 0.3300]
y_base2 = [0.4780, 0.4815, 0.4710, 0.4590, 0.4600, 0.4630]

plt.plot(X, y_LR, label='Logistic Regression')
plt.plot(X, y_NB1, label='Naive Bayes')
#plt.plot(X, y_NB2, label='Naive Bayes 2')
plt.plot(X, y_SVM1, label='Suppor Vector Machine')
#plt.plot(X, y_SVM2, label='Suppor Vector Machine 2')
plt.plot(X, y_base1, label='Baseline 1', linestyle='-')
plt.plot(X, y_base2, label='Baseline 2', linestyle='-')
plt.legend()
plt.show()