import sys
sys.path.append('GeLT')

import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from GeLT import GeLT
import re
import json
import numpy as np
import pandas as pd
import preprocess
from pytorch_pretrained_bert import BertTokenizer
from Transfomer_encoder import transfomer_encoder
from settings import benchmark_settings

datasets = "Android,Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Windows,Zookeeper"
datasets = datasets.split(",")
batch_size = 100
use_full = False
data_type = "2k"


def is_number(s):
    for func in (float, lambda x: int(x, 0)):
        try:
            func(s)
            return True
        except ValueError:
            continue
    return False

def read_groundtruth(logdir, dataset, use_full=False):
    data_type = 'full' if use_full else '2k'
    logpath = logdir / dataset / "{}_{}.log_structured{}.csv".format(dataset,data_type,"_corrected" if not use_full else "")
    df = pd.read_csv(logpath)
    return df

def add_true_values(dataset,workdir,probpath):
    logdir = workdir / "logs"
    df = read_groundtruth(logdir, dataset)
    with open(probpath / f'{dataset}.json', 'r') as file:
        word_probs = json.load(file) #['values']

    assert len(word_probs) == len(df), f"{len(word_probs)} == {len(df)}"

    word_is_const = []
    for i,(logmsg,template) in enumerate(df[['Content','EventTemplate']].values.tolist()):
        # print(i)
        logmsg_split = preprocess.wordsplit(logmsg.strip(),dataset)
        templt_split = preprocess.wordsplit(template.strip(),dataset)
        # print(logmsg_split,templt_split)
        words = word_probs[i]['words']
        is_const = [0 if e in template else 1 for e in words]
        word_is_const.append(is_const)

        assert(len(word_probs[i]['words']) == len(word_is_const[i]))
        # if len(word_probs[i]['words']) != len(word_is_const[i]):
        #     print('x')
        #     print(word_probs[i])
        #     print(word_is_const[i])
        #     exit()
        word_probs[i].update({"y_true": word_is_const[i]})
        # word_probs[i]['probs'] = [1-e for e in word_probs[i]['probs']]
    with open(probpath / f'{dataset}1.json', 'w') as outfile:
        json.dump(word_probs, outfile, indent=4)
    return word_probs

def analyze_probabilities(probpath, dataset):
        # if dataset!='HealthApp': continue
        with open(probpath / f'{dataset}1.json', 'r') as file:
            word_probs = json.load(file)
            probs = [ e['probs'] for e in word_probs ]
            probs_origin = [ e['probs_origin'] for e in word_probs ]
            y_true = [ e['y_true'] for e in word_probs ]
            # print(y_true[1])
            # print(probs[1])
            # print(probs_origin[1])
            cons, vars = list(), list()
            cons_origin, vars_origin = list(), list()
            for row_true, row_probs, row_origin in zip(y_true, probs, probs_origin):
                for t, u, v in zip(row_true, row_probs, row_origin):
                    if t == 1:
                        vars.append(u)
                        vars_origin.append(v)
                    elif t == 0:
                        cons.append(u)
                        cons_origin.append(v)

            mean_cons = sum(cons) / len(cons)
            mean_vars = sum(vars) / len(vars)
            mean_cons_origin = sum(cons_origin) / len(cons_origin)
            mean_vars_origin = sum(vars_origin) / len(vars_origin)

            # print(dataset, mean_vars, mean_cons, mean_vars - mean_cons,
            #       mean_vars_origin, mean_cons_origin, mean_vars_origin - mean_cons_origin)
            print(dataset, mean_vars - mean_cons, mean_vars_origin - mean_cons_origin)


def read_csv_to_list(file_path):
    data_list = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data_list.append(row[0])
    return data_list


models = [ 'log3t'] #, 'log3t-ttt' ]


if __name__ == '__main__':

    workdir = Path('.')
    logdir = Path('/local/home/enan/projects/loghub-2.0/')
    probpath = workdir / "probabilities5"
    probpath = workdir / "probabilities-16"

    for dataset in datasets:
        # if dataset!='HDFS': continue
        # if dataset not in ['HDFS', 'Hadoop', 'Spark', 'Zookeeper', 'BGL', 'HPC', 'Thunderbird', 'Proxifier', 'Windows', 'Linux' ]: continue
        add_true_values(dataset,workdir,probpath)
        analyze_probabilities(probpath, dataset)

    for model in models:
        n = 5 + 1
        # fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), sharey=True, layout="constrained")
        fig, ax = plt.subplots(nrows=2, ncols=8, figsize=(16, 6), sharey=True, layout="constrained")
        for figidx, dataset in enumerate(datasets):
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

    logdir = Path('/local/home/enan/projects/loghub-2.0/{}_dataset'.format(data_type))
    for dataset, setting in benchmark_settings.items():
        if use_full and dataset in [ 'Android','Windows', 'BGL','HDFS','Spark','Thunderbird' ]:
            continue
        # if dataset!='Apache':continue
    
        print(dataset)
        model=transfomer_encoder.BERT()
        model2=transfomer_encoder.BERT()
        model3=transfomer_encoder.BERT()
        start=datetime.datetime.now()
        fromated_log = preprocess.format_log(
            log_format=setting['log_format'],
            indir = str(logdir / dataset),
        )
        form = fromated_log.format('{}_{}.log'.format(dataset,data_type))
        log_content = form['Content'].to_numpy().tolist()
        if use_full:
            ground_file = logdir / dataset / '{}_{}.log_structured.csv'.format(dataset,data_type)
        else:
            ground_file = logdir / dataset / '{}_{}.log_structured_corrected.csv'.format(dataset,data_type)
        ground_truth_read=pd.read_csv(ground_file)
        ground_truth_list=list(ground_truth_read['EventTemplate'])
        variablelist=GeLT.read_csv_to_list('Variableset/variablelist1'+dataset+'.csv')
        constantlist=GeLT.read_csv_to_list('Variableset/constantlist'+dataset+'.csv')
        parse_data,log_sentence=GeLT.log_to_model(log_content,stage='parse',regx=[],regx_use=False,dataset=dataset,)
        train_data,_=GeLT.log_to_model(log_content,stage='train',regx=setting['regex'],regx_use=False,
                                            dataset=dataset,variablelist=variablelist,constantlist=constantlist)
    
        incorrect, correct = 0, 0
        numbers = 0
        tokenizer = BertTokenizer.from_pretrained('Vocab/Vocab.txt')
        tokens_list = []
        labels_list = []
        j = 0
        for i in range(len(train_data)):
            if i>0 and i%3==0: j+=1
            content, template = log_content[j], ground_truth_list[j]
            clean_tokens = []
            clean_labels = []
            tokens = tokenizer.convert_ids_to_tokens(train_data[i][0])
            labels = train_data[i][1]
            assert len(tokens) == len(labels)
            for t,l in zip(tokens,labels):
                if t.startswith("##"):
                    if l == 1:
                        clean_labels[-1] = l
                    # else:
                    # clean_tokens.append(t)
                    clean_tokens[-1] = clean_tokens[-1] + t[2:]
                elif t == '[PAD]':
                    continue
                else:
                    clean_tokens.append(t)
                    clean_labels.append(l)
            assert len(clean_tokens) == len(clean_labels), print(f"{len(clean_tokens)} == {len(clean_labels)}")
            tokens_list.append(clean_tokens)
            labels_list.append(clean_labels)
            for token, label in zip(clean_tokens, clean_labels):
                if label == 0:
                    if token in template:
                        correct+=1
                    else:
                        incorrect+=1
                    if GeLT.is_number(token):
                        numbers+=1
    
        total = correct+incorrect
        print(f"{dataset} {(incorrect/total)*100}" )
        print(f"{dataset} {(numbers/total)*100}" )
        print(f"{dataset} {(numbers/incorrect)*100}" )
