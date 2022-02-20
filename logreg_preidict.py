import pandas as pd
import numpy as np
import pickle


class Predictor:
    def __init__(self, df_test, df_thetas):
        self.df_test = df_test
        self.thetas = thetas

    @staticmethod
    def scaller(z):
        return (z - z.mean(axis=0)) / z.std(axis=0)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def prepare_prediction(self):
        self.df_test.fillna(0, inplace=True)
        x = self.df_test.iloc[:, 5:]
        # x = np.hstack((self.scaller(x.values), np.array([1]*x.shape[0]).reshape(-1,1)))
        x = self.scaller(x.values)
        result = []
        for y in self.thetas:
            z = self.sigmoid(x @ y)
            result.append(z)
        return self.predict(result)

    @staticmethod
    def reshape(matrix, r, c):
        matrix_n = [[0 for col in range(c)] for row in range(r)]
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix_n[j][i] = matrix[i][j]
        return matrix_n

    def predict(self, vector):
        HOUSES = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
        vector = self.reshape(vector, len(vector[0]), len(vector))
        result = []
        for i in range(len(vector)):
            maximum = -1
            index = 0
            for j in range(len(vector[i])):
                if vector[i][j] > maximum:
                    maximum = vector[i][j]
                    index = j
            result.append(HOUSES[index])
        return result


if __name__ == '__main__':
    try:
        df_test = pd.read_csv('datasets/dataset_test.csv', index_col='Index')
        with open('theta', 'rb') as f:
            thetas = pickle.load(f)
        predictor = Predictor(df_test, np.array(thetas))
        pred = predictor.prepare_prediction()
        pred = pd.DataFrame(pred, columns=['Hogwarts House'])
        pred.to_csv('predict_house.csv')
    except Exception as e:
        print(e)
