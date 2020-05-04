import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import json
import csv
import os

class PredictCarPrice():
    def __init__(self, path_of_data="programming_exercies/exercise/car/train_data/car_price.csv"):

        self.data =  pd.read_csv(path_of_data, sep='\t')


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.iloc[:, :-1],
                                                    self.data.iloc[:, -1],
                                                    test_size = 0.37,
                                                    random_state = 42)

        # self.linearRegression = LinearRegression()
        # self.model = self.linearRegression.fit(self.data.iloc[:, :-1], self.data.iloc[:, -1])


        self.model = LinearRegression().fit(self.data.iloc[:, :-1],
                                                    self.data.iloc[:, -1])


    def range_of_answer(self, list_of_answer):
        """
        get the range of answer (error 10%)
        :param list_of_answer: the list of answer (list)
        :return: list of range answer (list)
        """
        list_of_range_answer = []
        for a in list_of_answer:
            min_a = a - 0.05*a
            max_a = a + 0.05*a
            answer = [min_a, max_a]
            round_answer = [round(num, 3) for num in answer]
            list_of_range_answer.append(round_answer)
        return list_of_range_answer

    def filter_overlap_answer(self, list_of_range_answer):
        """
        filter overlap answer
        :param list_of_range_answer: the list of range answer (list)
        :return: the overlap of range answer (boolean)
        """
        compare = []
        for i in range(0, len(list_of_range_answer)-1):
            r = list_of_range_answer[0]
            w = list_of_range_answer[i+1]
            if w[1] < r[0] or w[0] > r[1]:
                compare.append('True')
            else:
                compare.append('False')

        if 'False' in compare:
            return False
        else:
            return True



    def generate_exercise(self):


        question1 = "The prices of new cars in the industry is fixed by the manufacturer with some additional costs incurred by the Government in the form of taxes. " \
                   "So, customers buying a new car can be assured of the money they invest to be worthy. " \
                   "But due to the increased price of new cars and the incapability of customers to buy new cars due to the lack of funds, " \
                   "used cars sales are on a global increase. There is a need for a used car price prediction system to effectively determine " \
                   "the worthiness of the car using a variety of features. Even though there are websites that offers this service, " \
                   "their prediction method may not be the best. Besides, different models and systems may contribute on predicting " \
                   "power for a used car’s actual market value. It is important to know their actual market value while both buying " \
                   "and selling. By using Linear regression, please predict the price of the used car with the parameters in X_dataset " \
                    "with R_square = 0.77"

        # question2 = "What is the value of R-square score if the training dataset is 0.667 and testing dataset is 0.333 of the data?"

        y_pred =  pd.Series(self.model.predict(self.X_test))

        answer = pd.DataFrame(y_pred)
        for i in range (1,5):
            noise = pd.DataFrame(np.random.normal(100, 1000, y_pred.shape[0]))

            wrong_answer = noise+pd.DataFrame(y_pred)
            answer = pd.DataFrame(pd.concat([answer, pd.DataFrame(wrong_answer)], axis=1, join='inner'))


        field_columns = ['right answer', 'wrong answer 1', 'wrong answer 2', 'wrong answer 3', 'wrong answer 4']
        answer.columns = field_columns
        solution = []
        index = []
        for k in range(0, len(answer)-1):
            if answer.loc[k]['right answer'] < 0 or answer.loc[k][ 'wrong answer 1'] < 0  or answer.loc[k]['wrong answer 2'] < 0 or answer.loc[k]['wrong answer 3']< 0 or answer.loc[k]['wrong answer 4']<0:
                continue
            else:
                range_of_answer = self.range_of_answer(answer.loc[k])
                if self.filter_overlap_answer(range_of_answer):
                    solution.append(range_of_answer)
                    X_test = pd.DataFrame(self.X_test.iloc[k]).T
                    index.append(k)

                    X_test.to_csv('./programming_exercies/exercise/car/test_data/' + 'car_' + str(k) + '.csv', sep='\t', header=True, index=None)
        solution = pd.DataFrame(solution)
        question_of_ex1 = []

        for i in range(len(solution)):
            question_of_ex1.append(question1)

        exercise = pd.DataFrame(pd.concat([solution, pd.DataFrame(question_of_ex1)], axis=1, join='inner'))
        field_columns = ['right answer', 'wrong answer 1', 'wrong answer 2', 'wrong answer 3', 'wrong answer 4', 'question']

        exercise.columns = field_columns

        for w in range(0, pd.DataFrame(solution).shape[0]-1):
            content = {}
            ex = pd.DataFrame(exercise.iloc[w]).T
            ex.insert(6, 'test_data', 'https://github.com/dungkorbit/fpt/blob/master/programming_exercies/exercise/car/test_data/'+'car_'+str(index[w])+'.csv')
            ex.insert(7, 'training_data', 'https://github.com/dungkorbit/fpt/blob/master/programming_exercies/exercise/car/train_data/car_price.csv')

            ex.to_csv('./programming_exercies/exercise/car/exercise/'+'exercise_car'+str(index[w])+'.csv', sep='\t',header=True, index=None)

            content['right_answer'] = ex['right answer'].values[0]

            w1 = ex['wrong answer 1'].values[0]
            w2 = ex['wrong answer 2'].values[0]
            w3 = ex['wrong answer 3'].values[0]
            w4 = ex['wrong answer 4'].values[0]

            content['wrong_answer'] = [w1,w2,w3,w4]
            content['question'] = ex['question'].values[0]
            content['test_data'] = ex['test_data'].values[0]
            content['train_data']= ex['training_data'].values[0]
            print(w)
            with open('./programming_exercies/car_edit.json', 'a+') as json_file:
                json.dump(content, json_file)




if __name__ == "__main__":
    predictprice = PredictCarPrice()
    predictprice.generate_exercise()
