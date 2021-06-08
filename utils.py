
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def savePlot(filename, x, y, english=True):
    line = [min(y), max(y)]

    plt.clf()

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    fig, ax = plt.subplots()
    if english:
      ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))
      ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))
    else:
      ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x).replace('.', ',')))
      ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x).replace('.', ',')))

    ax.scatter(x, y, color='red')
    ax.plot(line, line, color='blue', linewidth=3)

    ax.set_xlabel('Surrogate model estimate (billion USD)' if english else 'Estimativa do modelo auxiliar (US$ bi)', fontsize=15)
    ax.set_ylabel('Simulator output (billion USD)' if english else 'Saída do simulador (US$ bi)', fontsize=15)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)


def saveCsv(filename, prediction, original):
    prediction = pd.DataFrame(prediction, original)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    prediction.to_csv(filename + '.csv', decimal='.', sep=';')

