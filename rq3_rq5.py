import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import os
import json,csv
from scipy.interpolate import interp1d

def read_csv_to_list(file_path):
    data_list = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data_list.append(row[0])
    return data_list

workdir = Path(".")
models = [ 'log3t']

for model in models:
    n = 5 + 1
    # fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), sharey=True, layout="constrained")
    fig, ax = plt.subplots(nrows=2, ncols=8, figsize=(16, 6), sharey=True, layout="constrained")
    for figidx, dataset in enumerate(projects):
        # if dataset == 'HealthApp': continue
        boxplotlists = {}
        for Z in range(2):
            # ground_truth_list = pd.read_csv(workdir / 'logs' / dataset / (dataset+"_2k.log_structured_corrected.csv"))['EventTemplate']
            ground_truth_list = pd.read_csv(Path('/local/home/enan/projects/loghub-2.0/2k_dataset') / dataset / (dataset+"_2k.log_structured_corrected.csv"))['EventTemplate']
            # variablelist = read_csv_to_list(workdir/'Variableset'/ f'variablelist1{dataset}.csv')
            # with open(workdir / ("probabilities-"+model) / dataset / 'probabilities.json', 'r') as file:
            with open(workdir / f"probabilities{n}" / f'{dataset}1.json', 'r') as file:
                word_probs = json.load(file) #['values']
            # print(word_probs[0])
            if Z == 1:
                K = 'probs'
            # elif Z == 1:
            #     K = 'probs_withoutTTT'
            else:
                K = 'probs_origin' # 'probs_withoutTTT'
            probs = [ elem[K] for elem in word_probs]
            constants = [
                word_probs[i][K][j]
                for i in range(len(word_probs)) for j in range(len(word_probs[i]['words']))
                if word_probs[i]['y_true'][j] == 0
            ]
            variables = [
                word_probs[i][K][j]
                for i in range(len(word_probs)) for j in range(len(word_probs[i]['words']))
                if word_probs[i]['y_true'][j] == 1
            ]
            # unseenvars = [
            #     word_probs[i][K][j]
            #     for i in range(len(word_probs)) for j in range(len(word_probs[i]['words']))
            #     if word_probs[i]['y_true'][j] == 1 and word_probs[i]['words'][j] not in " ".join(variablelist)
            # ]
            boxplotlists[K] = [constants, variables] #, unseenvars])
        boxplotlists = [boxplotlists['probs_origin'][0],boxplotlists['probs'][0],boxplotlists['probs_origin'][1],boxplotlists['probs'][1]]
        # boxplotlists = [boxplotlists[0],boxplotlists[3],boxplotlists[1],boxplotlists[4],boxplotlists[2],boxplotlists[5]]
        # boxplotlists = [boxplotlists[0],boxplotlists[3],boxplotlists[6],boxplotlists[1],boxplotlists[4],boxplotlists[7],boxplotlists[2],boxplotlists[5],boxplotlists[8]]
        factor = 4*2
        fig_i, fig_j = figidx // factor, figidx % factor
        _fig = ax[fig_i][fig_j]
        _fig.boxplot(boxplotlists, showfliers=False)
        _fig.set_xticks([e+0.5 for e in list(range(1,5,2))],['constants','variables'])
        # _fig.set_xticks(range(2,7,2),['constants','variables','unseen'])
        # _fig.set_xticks(range(2,10,3),['constants','variables','unseen'])
        _fig.set_title(dataset)
    # plt.suptitle("Probabilities with and without Test-time Training",y=0.94)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(f'figures/probs-{n}-{model}.pdf', bbox_inches='tight', pad_inches=0)
