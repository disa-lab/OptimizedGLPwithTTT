import sys
sys.path.append('GeLT')

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import preprocess
from settings import benchmark_settings

datasets = "Android,Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Windows,Zookeeper"
datasets = datasets.split(",")
batch_size = 100


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

def get_probability_metrics(word_probs):
    correct_bin_0 = []
    incorrect_bin_0 = []
    correct_bin_1 = []
    incorrect_bin_1 = []

    n_words = [ len(w['words']) for w in word_probs ]
    n_words_correct_0 = [
        len(w['words']) for w in word_probs if (w['y_pred']==0 and w['y_pred']==w['y_true'])
    ]
    n_words_incorrect_0 = [
        len(w['words']) for w in word_probs if (w['y_pred']==0 and w['y_pred']!=w['y_true'])
    ]
    n_words_correct_1 = [
        len(w['words']) for w in word_probs if (w['y_pred']==1 and w['y_pred']==w['y_true'])
    ]
    n_words_incorrect_1 = [
        len(w['words']) for w in word_probs if (w['y_pred']==1 and w['y_pred']!=w['y_true'])
    ]

    for i in range(len(word_probs)):
        _words = word_probs[i]['words']
        y_prob = word_probs[i]['probs']
        y_pred = word_probs[i]['y_pred']
        y_true = word_probs[i]['y_true']

        if not (len(_words) == len(y_prob) == len(y_pred) == len(y_true)):
            print(i)
            print(_words)
            print(len(_words) , len(y_prob) , len(y_pred) , len(y_true))
        assert(len(_words) == len(y_prob) == len(y_pred) == len(y_true))

        for j in range(len(y_prob)):
            # if is_number(_words[j]): continue
            if y_pred[j] == 1:
                if y_true[j] == y_pred[j]:
                    correct_bin_1.append(y_prob[j])
                elif y_true[j] != y_pred[j]:
                    incorrect_bin_1.append(y_prob[j])
            else:
                if y_true[j] == y_pred[j]:
                    correct_bin_0.append(y_prob[j])
                elif y_true[j] != y_pred[j]:
                    incorrect_bin_0.append(y_prob[j])

    return (
        n_words,
        n_words_correct_0,
        n_words_incorrect_0,
        correct_bin_0,
        incorrect_bin_0,
        n_words_correct_1,
        n_words_incorrect_1,
        correct_bin_1,
        incorrect_bin_1,
    )


def analyze_probability(workdir,probpath):

    correct_bin_0_mean_dict = {}
    correct_bin_0_stdv_dict = {}
    incorrect_bin_0_mean_dict = {}
    incorrect_bin_0_stdv_dict = {}
    correct_bin_1_mean_dict = {}
    correct_bin_1_stdv_dict = {}
    incorrect_bin_1_mean_dict = {}
    incorrect_bin_1_stdv_dict = {}

    for dataset in datasets:
        # if dataset != 'HealthApp': continue
        print(dataset)

        word_probs = add_true_values(dataset,workdir,probpath)
        (
            n_words,
            n_words_correct_0,
            n_words_incorrect_0,
            correct_bin_0,
            incorrect_bin_0,
            n_words_correct_1,
            n_words_incorrect_1,
            correct_bin_1,
            incorrect_bin_1,
        ) = get_probability_metrics(word_probs)

        correct_bin_0_mean   = np.mean(correct_bin_0)
        correct_bin_0_stdv   = np.std( correct_bin_0)
        incorrect_bin_0_mean = np.mean(incorrect_bin_0)
        incorrect_bin_0_stdv = np.std( incorrect_bin_0)

        correct_bin_1_mean   = np.mean(correct_bin_1)
        correct_bin_1_stdv   = np.std( correct_bin_1)
        incorrect_bin_1_mean = np.mean(incorrect_bin_1)
        incorrect_bin_1_stdv = np.std( incorrect_bin_1)

        # print(f"#Words:         {np.mean(n_words)}")
        # print(f"Threshold:      {benchmark_settings[dataset]['threshold']}")
        # print(f"Correct 0:      {correct_bin_0_mean:0.04f} - {correct_bin_0_stdv:0.04f}")
        # print(f"Correct 1:      {correct_bin_1_mean:0.04f} - {correct_bin_1_stdv:0.04f}")
        # print(f"Incorrect 0:    {incorrect_bin_0_mean:0.04f} - {incorrect_bin_0_stdv:0.04f}")
        # print(f"Incorrect 1:    {incorrect_bin_1_mean:0.04f} - {incorrect_bin_1_stdv:0.04f}")
        # print()

        correct_bin_0_mean_dict[dataset] =          correct_bin_0_mean
        correct_bin_0_mean_dict[dataset] =          correct_bin_0_mean
        correct_bin_0_stdv_dict[dataset] =          correct_bin_0_stdv
        incorrect_bin_0_mean_dict[dataset] =        incorrect_bin_0_mean
        incorrect_bin_0_stdv_dict[dataset] =        incorrect_bin_0_stdv
        correct_bin_1_mean_dict[dataset] =          correct_bin_1_mean
        correct_bin_1_stdv_dict[dataset] =          correct_bin_1_stdv
        incorrect_bin_1_mean_dict[dataset] =        incorrect_bin_1_mean
        incorrect_bin_1_stdv_dict[dataset] =        incorrect_bin_1_stdv

    return (
        correct_bin_0_mean_dict,
        correct_bin_0_stdv_dict,
        incorrect_bin_0_mean_dict,
        incorrect_bin_0_stdv_dict,
        correct_bin_1_mean_dict,
        correct_bin_1_stdv_dict,
        incorrect_bin_1_mean_dict,
        incorrect_bin_1_stdv_dict,
    )

def analyze_probability_all_models(workdir):
    models = ['log3t','tfbert-2','roberta-2',]
    df_list = []
    for model in models:
        probpath = workdir / ("probabilities-"+model)
        (
            correct_bin_0_mean_dict,
            correct_bin_0_stdv_dict,
            incorrect_bin_0_mean_dict,
            incorrect_bin_0_stdv_dict,
            correct_bin_1_mean_dict,
            correct_bin_1_stdv_dict,
            incorrect_bin_1_mean_dict,
            incorrect_bin_1_stdv_dict,
        ) = analyze_probability(workdir,probpath)
        # index = ['Correct_0','Correct_1','Incorrect_0','Incorrect_1']
        index = ['C0','C1','I0','I1']
        index = [ind+'_'+model.replace('-2','') for ind in index]
        df = pd.DataFrame([correct_bin_0_mean_dict, incorrect_bin_0_mean_dict,
                           correct_bin_1_mean_dict, incorrect_bin_1_mean_dict],
                          index=index)
        df = df.transpose()
        # print(df)
        df_list.append(df)
    df = pd.concat(df_list,axis=1)
    df.sort_index(axis=1, inplace=True)
    df.to_csv("analysis.csv", float_format="%.3f")



def num_there(s):
    digits = [i.isdigit() for i in s]
    return True if np.mean(digits) >0.0 else False

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

            print(dataset, mean_vars, mean_cons, mean_vars - mean_cons,
                  mean_vars_origin, mean_cons_origin, mean_vars_origin - mean_cons_origin)

if __name__ == '__main__':

    workdir = Path('.')
    logdir = Path('/local/home/enan/projects/loghub-2.0/')
    probpath = workdir / "probabilities5"
    probpath = workdir / "probabilities-16"

    type = '2k'

    # analyze_probability_all_models(workdir)

    for dataset in datasets:
        # if dataset!='HDFS': continue
        # if dataset not in ['HDFS', 'Hadoop', 'Spark', 'Zookeeper', 'BGL', 'HPC', 'Thunderbird', 'Proxifier', 'Windows', 'Linux' ]: continue
        add_true_values(dataset,workdir,probpath)
        analyze_probabilities(probpath, dataset)

    # analyze_probability(workdir,probpath)
    # analyze_templates(logdir / "2k_dataset")
    # analyze_differences(logdir / f"{type}_dataset", resultdir)

    # resultdir1 = Path('/local/home/enan/projects/loghub-2.0/result/result_LogPPT-orig_full/')
    # resultdir2 = Path('/local/home/enan/projects/loghub-2.0/result/result_GeLL-Drain_full/')
    # resultdir1 = Path('/local/home/enan/projects/loghub-2.0/result/result_LILAC_full/')
    # resultdir2 = Path('/local/home/enan/projects/loghub-2.0/result/result_SynLogPlus-LILAC_full/')
    # compare_parsed_logs(logdir / f"{type}_dataset", resultdir1, resultdir2)
