import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def pair_plot(df):
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(df, hue="Hogwarts House");
    os.system('mkdir pics')
    plt.savefig('pics/pair_plot.png')
    plt.show()



if __name__ == '__main__':
    df = pd.read_csv('datasets/dataset_train.csv')
    df['Hogwarts House'].value_counts()
    pair_plot(df)