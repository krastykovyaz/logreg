import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class Coach:
    def __init__(self, df_train):
        self.df = df_train

    @staticmethod
    def scaller(z):
        return (z - z.mean(axis=0)) / z.std(axis=0)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def plot(losses, houses):
        plt.figure(figsize=(14, 8))
        for i, house in enumerate(losses):
            sns.scatterplot(x=range(len(house)), y=house, label=houses[i])
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        os.system('mkdir pics')
        plt.savefig('pics/training.png')


    def get_thetas(self, df):
        HOUSES = self.df['Hogwarts House']
        MARK = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
        df.fillna(0, inplace=True)
        x = df.iloc[:, 5:]
        print(x.shape)
        # x = np.hstack((self.scaller(x.values), np.array([1]*df.shape[0]).reshape(-1,1)))
        x = self.scaller(x.values)
        print(x.shape)
        y = []
        for m in range(len(MARK)):
            y.append(np.array(MARK[m] == HOUSES).astype(int))
        return x, y, MARK


    def gradient_descent(self, x, y, epochs=10, lr=0.01, eps=0.0001):
        losses = []
        prev_loss = 1
        size = len(x)
        theta = np.zeros((1, x.shape[1]))
        i = 0
        while i < epochs:
            print('f')
            print(theta.shape, x.T.shape)
            p = self.sigmoid(theta @ x.T)
            print('f2')
            grad = (p - y.T) @ x / size
            loss = np.sum((np.log(p) * y.T) + (np.log(1 - p) * (1 - y.T))) / -size
            losses.append(loss)
            i += 1
            if abs(prev_loss - loss) < eps:
                break
            theta -= lr * grad
            prev_loss = loss
        print(f'There were {i} epochs !')
        return theta[0].tolist(), losses


    def train(self, x, y):
        losses = []
        thetas = []
        for i in range(len(y)):
            theta, loss = self.gradient_descent(x, y[i])
            losses.append(loss)
            thetas.append(theta)
        return thetas, losses


if __name__ == '__main__':
    try:
        df_train = pd.read_csv('datasets/dataset_train.csv', index_col='Index')
        coach = Coach(df_train)
        x, y, houses = coach.get_thetas(df_train)
        thetas, losses = coach.train(x, y)
        coach.plot(losses, houses)
        with open('theta', 'wb') as f:
            pickle.dump(thetas, f)
    except Exception as e:
        print(e)
