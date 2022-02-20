import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import os
import numpy as np


def get_grades(dataset, prep_dataset, house, topic):
    return prep_dataset[dataset["Hogwarts House"] == house][topic]


def scatter_plot(df):
    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    fig, axes = plt.subplots(14, 14, figsize=(60, 40))
    houses = [house for house in df['Hogwarts House'].value_counts().index]
    labels = [house[:3] for house in houses]
    colors = np.random.choice(list(mcolors.BASE_COLORS.keys()), len(labels))
    i = 1
    for col in df_numeric.columns:
        for col2 in df_numeric.columns:
            i += 1
            try:
                plt.subplot(14, 14, i)
                for j, house in enumerate(houses):
                    sns.scatterplot(data=df[df["Hogwarts House"] == house], x=col, \
                                    y=col2, \
                                    label=labels[j], \
                                    color=colors[j])
                plt.legend(loc='upper left')
                plt.title(col)
            except:
                break
    fig.tight_layout()
    os.system('mkdir pics')
    plt.savefig('pics/scatter_plot.png')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('datasets/dataset_train.csv')
    df['Hogwarts House'].value_counts()
    scatter_plot(df)
