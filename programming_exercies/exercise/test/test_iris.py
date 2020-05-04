
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import random

data = pd.read_csv("train.csv", sep='\t')
test = pd.read_csv("test.csv", sep='\t')

X_train = pd.DataFrame(data.loc[:, 'petal.length'].values)
y_train = pd.DataFrame(data.iloc[:, len(data.columns) - 1].values)

X_test = pd.DataFrame(test.loc[:, 'petal.length'].values)
y_test = pd.DataFrame(test.iloc[:, len(test.columns) - 1].values)
acc = []
f1c = []
for i in range(0, 1000):
    random.seed(i)
    model = LogisticRegression(random_state=i).fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracry = metrics.accuracy_score(prediction, y_test)
    f1 = metrics.f1_score(prediction, y_test, average='micro')
    acc.append(accuracry)
    f1c.append(f1)

print("Accuracy = ")
print(min(acc), max(acc))
print('F1-score = ')
print(min(f1c), max(f1c))