
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import random
import numpy

data = pd.read_csv("train.csv", sep='\t')
test = pd.read_csv("test.csv", sep='\t')
numpy.random.seed()
X_train = pd.DataFrame(data.loc[:, 'sepal.length'].values)
y_train = pd.DataFrame(data.iloc[:, len(data.columns) - 1].values)

X_test = pd.DataFrame(test.loc[:, 'sepal.length'].values)
y_test = pd.DataFrame(test.iloc[:, len(test.columns) - 1].values)


model = LogisticRegression().fit(X_train, y_train)

prediction = model.predict(X_test)
accuracry = metrics.accuracy_score(prediction, y_test)
f1 = metrics.f1_score(prediction, y_test, average='micro')

print(accuracry, f1)