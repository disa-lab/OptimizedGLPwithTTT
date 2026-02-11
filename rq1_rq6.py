import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import os
import json,csv
from scipy.interpolate import interp1d

rootdir = "."
parsers = [
    [rootdir+"/GeLTTT/output/Log3T/probs/",    "{}_2k.log_structured_result.csv"],
    [rootdir+"/GeLTTT/output/origin/probs/", "{}_2k.log_structured_result.csv"],
    [rootdir+"/GeLTTT/output/GeLT/probs/",    "{}_2k.log_structured_result.csv"],
]
legs = [
    'With TTT',
    'Without TTT',
    'With Optimized TTT',
]

metrics = ['GA', 'PA', 'FGA','FTA']
prefix = "T5"
prefix = "Log3T"
for metric in metrics:
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 12), layout="constrained")
    
    i = 0
    for row in ax:
        for col in row:
            if i >= len(projects):
                break
            system = projects[i]
            i += 1
            for j in range(len(parsers)):
                filepath = parsers[j][0] + parsers[j][1].replace("result","result").format(system)
                # print(filepath)
                if not Path(filepath).exists(): continue
                df = pd.read_csv( filepath, sep=",")
                _line, = col.plot(df['Batch'],df[metric], linestyle='dotted',linewidth=(3-(1*j/len(parsers))), alpha=(1-0.2*j if j==1 else 1))
                # _line.set_dashes([2+j*2, 5])

            col.legend(legs)
            # col.grid()
            col.set_title(system)
    if True:
        plt.suptitle(f"{prefix}-{metric}", y=0.94)
    else:
        # plt.savefig(f"figures/{prefix}-{metric}.pdf", bbox_inches='tight', pad_inches=0.5)
        plt.savefig(f"figures/{prefix}1-{metric}.pdf", bbox_inches='tight', pad_inches=0.5)
