import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os


def get_grades(dataset, prep_dataset, house, topic):
    df = prep_dataset[dataset["Hogwarts House"] == house][topic]
    return df.dropna()


def histogram(df):
    df_numeric = df.select_dtypes(include=['int64', 'float64'])

    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    kwargs = dict(bins=30, alpha=0.3)
    houses = [house for house in df['Hogwarts House'].value_counts().index]
    labels = [house[:3] for house in houses]
    colors = np.random.choice(list(mcolors.BASE_COLORS.keys()), len(labels))
    for i, col in enumerate(df_numeric.columns):
        plt.subplot(5, 3, i + 1)
        for j, house in enumerate(houses):
            sns.histplot(df[df["Hogwarts House"] == house][col], \
                         label=labels[j], \
                         color=colors[j], \
                         **kwargs)
        plt.legend(loc='upper left')
        plt.title(col)
    fig.tight_layout()
    os.system('mkdir pics')
    plt.savefig('pics/histograms.png')
    plt.show()



if __name__ == '__main__':
    df = pd.read_csv('datasets/dataset_train.csv')
    df['Hogwarts House'].value_counts()
    histogram(df)
