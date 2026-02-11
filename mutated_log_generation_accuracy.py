import datetime
from Log3T import Log3T
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from Transfomer_encoder import transfomer_encoder
from settings import benchmark_settings
from Log3T import preprocess
import os
from pathlib import Path



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-full", "--use_full", action="store_true")
args = parser.parse_args()
type = 'full' if args.use_full else '2k'

logdir = Path('/local/home/enan/projects/loghub-2.0/{}_dataset'.format(type))

for dataset, setting in benchmark_settings.items():
    if args.use_full and dataset in [ 'Android','Windows', 'BGL','HDFS','Spark','Thunderbird' ]:
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
    form = fromated_log.format('{}_{}.log'.format(dataset,type))
    log_content = form['Content'].to_numpy().tolist()
    if args.use_full:
        ground_file = logdir / dataset / '{}_{}.log_structured.csv'.format(dataset,type)
    else:
        ground_file = logdir / dataset / '{}_{}.log_structured_corrected.csv'.format(dataset,type)
    ground_truth_read=pd.read_csv(ground_file)
    ground_truth_list=list(ground_truth_read['EventTemplate'])
    variablelist=Log3T.read_csv_to_list('Variableset/variablelist1'+dataset+'.csv')
    constantlist=Log3T.read_csv_to_list('Variableset/constantlist'+dataset+'.csv')
    parse_data,log_sentence=Log3T.log_to_model(log_content,stage='parse',regx=[],regx_use=False,dataset=dataset,)
    train_data,_=Log3T.log_to_model(log_content,stage='train',regx=setting['regex'],regx_use=False,
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
                if Log3T.is_number(token):
                    numbers+=1

    total = correct+incorrect
    print(f"{dataset} {(incorrect/total)*100}" )
    print(f"{dataset} {(numbers/total)*100}" )
    print(f"{dataset} {(numbers/incorrect)*100}" )
    continue
