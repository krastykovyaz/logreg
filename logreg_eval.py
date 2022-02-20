import pandas as pd


class Evaluator:
    def __init__(self):
        self.df_valid = pd.read_csv('datasets/truth_dataset.csv')
        self.df_pred = pd.read_csv('predict_house.csv')

    def calc(self):
        counter = 0
        for i in range(len(self.df_pred['Hogwarts House'])):
            if self.df_pred['Hogwarts House'][i] == self.df_valid['Hogwarts House'][i]:
                counter += 1
        share = counter / len(self.df_pred['Hogwarts House'])
        return share


if __name__ == '__main__':
    evaluator = Evaluator()
    print('Accuracy:', evaluator.calc())
