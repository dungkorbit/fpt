
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from os.path import basename
from pathlib import Path
import os
import json

class ProgrammingExcerciseGenerate():
    def __init__(self, path_of_dataset):
        """
        :param path_of_dataset: the path of considered dataset (string)
        """
        self.data = pd.read_csv(path_of_dataset, sep=",")
        self.name_dataset = Path(basename(path_of_dataset)).stem

    def range_of_answer(self, list_of_answer):
        """
        get the range of answer (error 10%)
        :param list_of_answer: the list of answer (list)
        :return: list of range answer (list)
        """
        list_of_range_answer = []
        for a in list_of_answer:
            min_a = a - 0.2*a
            max_a = a + 0.2*a
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


    def exercise_accuracy_LogisticRegression(self,name_of_generated_dataset, XX, yy, question, scenario):
        """
        :param name_of_generated_dataset: the name of dataset (string)
        :param XX: the considered X array of dataset (dataframe)
        :param yy: the considered y arracy of dataset (dataframe)
        :return answer: the list of right, wrong answers (list)
        """
        list_of_answer = []
        X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.30)

        model = LogisticRegression().fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracry = metrics.accuracy_score(prediction, y_test)
        f1 = metrics.f1_score(prediction, y_test, average='micro')


        train_data = pd.DataFrame(pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1, join='inner'))
        test_data = pd.DataFrame(pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1, join='inner'))
        field_name = scenario + ['variety']
        test_data.columns = field_name
        train_data.columns = field_name
        test_name = 'test_data_' + name_of_generated_dataset+ "_" + str(scenario) + ".csv"
        train_name = 'train_data_' + name_of_generated_dataset + "_" + str(scenario) + ".csv"

        context = "Please use Logistic Regression program to classify an iris species as either (virginica, setosa, or versicolor) " \
                  "then evaluate the accuracy and f1-score which was calculated by counting the total true positives " \
                  "false negatives and false positives of the algorithm given the training and testing dataset. "
        temp_answer = []
        if "accuracy" in question:
            right_answer = accuracry
            for i in range(0,4):
                temp_answer.append(right_answer)
        elif "f1-score" in question:
            right_answer = f1
            for i in range(0, 4):
                temp_answer.append(right_answer)

        noise = np.random.normal(0.1, 0.5, 4)
        wrong_answer = list(noise + np.array(temp_answer))
        list_of_answer.append(right_answer)
        list_of_answer.extend(wrong_answer)
        list_of_range_answer = self.range_of_answer(list_of_answer)
        test_name1 = 'https://github.com/dungkorbit/fpt/blob/master/programming_exercies/exercise/iris/test_data/' + test_name
        train_name1 = 'https://github.com/dungkorbit/fpt/blob/master/programming_exercies/exercise/iris/train_data/' + train_name

        if self.filter_overlap_answer(list_of_range_answer):
            list_of_range_answer.append(context.strip()+ question)
            list_of_range_answer.append(train_name1)
            list_of_range_answer.append(test_name1)
            test_data.to_csv(test_name, sep='\t', index=None)
            train_data.to_csv(train_name, sep='\t', index=None)
            return list_of_range_answer
        else:
            return False

    def sub_lists(self, list1):
        """
        get the sublist from a given list
        :param list1: the considered list (list)
        return: sublist: the list of sublist (list)
        """

        # store all the sublists
        sublist = [[]]

        # first loop
        for i in range(len(list1) + 1):

            # second loophttps://github.com/account/unverified-email
            for j in range(i + 1, len(list1) + 1):
                # slice the subarray
                sub = list1[i:j]
                sublist.append(sub)

        return sublist

    def generate_exercise(self, number_of_data, path):
        """
        :param number_of_data: the number of generated data (int)
        """

        field_name = ['sepal.length','sepal.width','petal.length','petal.width','variety']
        field_exercise = ['right_answer','wrong_answer_1', 'wrong_answer_2','wrong_answer_3','wrong_answer_4','question','train_dataset', 'test_dataset']
        fields = field_name[:-1]
        scenarios = self.sub_lists(fields)

        q1 = ' What is the f1-score of Logistic Regression which was calculated by counting the total true positives false negatives and false positives.?'
        q2 = ' What is the accuracy of Logistic Regression ?'
        X = self.data.iloc[:,:-1].values
        y = self.data.iloc[:,4].values
        os.chdir(path)
        data_frame = {}
        for i in range(1, number_of_data+1):
            e = np.random.randint(10, size=(len(pd.DataFrame(X)), len(pd.DataFrame(X).columns)))
            XX = X+e*0.5
            yy = y
            new_data = pd.DataFrame(pd.concat([pd.DataFrame(XX), pd.DataFrame(yy)], axis=1, join='inner'))
            new_data.columns = field_name
            for scenario in scenarios:
                if len(scenario) < 1:
                    continue
                else:
                    XX = new_data.loc[:, scenario].values
                    yy = new_data.iloc[:, len(new_data.columns)-1].values
                    # for j, question in enumerate(list_question):
                    answer1 = self.exercise_accuracy_LogisticRegression(str(i)+"_"+self.name_dataset+".csv",XX,yy, q1, scenario)
                    answer2 = self.exercise_accuracy_LogisticRegression(str(i)+"_"+self.name_dataset+".csv",XX,yy, q2, scenario)
                    if answer1 is False or answer2 is False:
                        continue
                    else:
                        print('exercises_' + str(i) + "_" + self.name_dataset + "_" + str(scenario))
                        answer_store = pd.DataFrame(zip(answer1, answer2)).T
                        answer_store.columns = field_exercise
                        answer_store.to_csv(
                            'exercises_' + str(i) + "_" + self.name_dataset + "_" + str(scenario) + ".csv",
                            sep='\t', header=None, index=None, mode='a')

                for w in range(0, len(answer_store)-1):
                    content = {}
                    content['right_answer'] = answer_store.loc[w][0]

                    w1 = answer_store.loc[w][1]
                    w2 = answer_store.loc[w][2]
                    w3 = answer_store.loc[w][3]
                    w4 = answer_store.loc[w][4]

                    content['wrong_answer'] = [w1, w2, w3, w4]
                    content['question'] =  answer_store.loc[w][5]
                    content['train_data'] = answer_store.loc[w][6]
                    content['test_data'] = answer_store.loc[w][7]

                    with open('../iris_edit.json', 'a+') as json_file:
                        json.dump(content, json_file)





if __name__ == "__main__":
    programmingexcercise = ProgrammingExcerciseGenerate("./programming_exercies/data/irris/iris.csv")
    programmingexcercise.generate_exercise(600,  "./programming_exercies/exercise/iris/")
