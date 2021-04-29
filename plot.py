import random
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def generate_random_color_names(k: int, seed: int):
    random.seed(seed)
    colors = []

    for name, _ in matplotlib.colors.cnames.items():
        colors.append(name)

    return random.choices(colors, k=k)


def plot_embedding(dataframe: pd.DataFrame, target_field: str, target_filter: Callable, out_file: str, title: Optional[str] = None):
    figure = plt.figure(figsize=(16, 16))

    ax = figure.add_subplot(1, 1, 1)
    ax.set_xlabel('x_axis', fontsize=15)
    ax.set_ylabel('y_axis', fontsize=15)

    if title:
        ax.set_title(title, fontsize=20)
    else:
        ax.set_title('', fontsize=20)

    targets = set(filter(target_filter, set(dataframe[target_field])))
    colors = generate_random_color_names(len(targets), 17)

    for i, target in enumerate(targets):
        idx_to_plot = dataframe[target_field] == target
        x_axis = dataframe.loc[idx_to_plot, 'Principal Component 1']
        y_axis = dataframe.loc[idx_to_plot, 'Principal Component 2']
        ax.scatter(x_axis, y_axis, c=colors[i], s=15)

    ax.legend(targets)
    ax.grid()
    plt.savefig(out_file)
