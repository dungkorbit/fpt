import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x_data = []
y_data = []

with open('programming_exercies/exercise/car/train_data/car_price.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    first = True
    for row in spamreader:
        if first:
            first = False
            continue
        x_data.append(row[0:-1]) # Skip first input, because it is the example number
        y_data.append(row[-1])


x_data = np.asarray(x_data, dtype='float')
y_data = np.asarray(y_data, dtype='float')
print(y_data)
print("x_data.shape", x_data.shape)
print("y_data.shape", y_data.shape)
reg = LinearRegression().fit(x_data, y_data)
y_pred = reg.predict(x_data)
print("r2_score", r2_score(y_data, y_pred))
#
x_test = []
with open('programming_exercies/exercise/test/car_62.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    first = True
    for row in spamreader:
        if first:
            first = False
            continue
        x_test.append(row)

x_test = np.asarray(x_test, dtype='float').T
print("x_test.shape", x_test.shape)
#
y_pred = reg.predict(x_test)
print("predicted price (y_pred)", y_pred)
