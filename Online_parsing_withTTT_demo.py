import datetime
from GeLT import GeLT
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from Transfomer_encoder import transfomer_encoder
from GeLT import preprocess
import os
from pathlib import Path

'''
model will be constantly update using test time training
model2 is used to load parameters trained with only first batch
model3 is used to load parameters trained with all logs
'''

from settings import benchmark_settings


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-full", "--use_full", action="store_true")
# parser.add_argument("-oput", "--output", default="output/ttt")
# parser.add_argument("-ttt", "--enable_ttt", action="store_true")
parser.add_argument("-optim", "--use_optimized", action="store_true")

args = parser.parse_args()
data_type = 'full' if args.use_full else '2k'

logdir = Path('/local/home/enan/projects/loghub-2.0/{}_dataset'.format(data_type))

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
    form = fromated_log.format('{}_{}.log'.format(dataset,data_type))
    log_content = form['Content'].to_numpy().tolist()
    if args.use_full:
        ground_file = logdir / dataset / '{}_{}.log_structured.csv'.format(dataset,data_type)
    else:
        ground_file = logdir / dataset / '{}_{}.log_structured_corrected.csv'.format(dataset,data_type)
    ground_truth_read=pd.read_csv(ground_file)
    ground_truth_list=list(ground_truth_read['EventTemplate'])
    variablelist=GeLT.read_csv_to_list('Variableset/variablelist1'+dataset+'.csv')
    constantlist=GeLT.read_csv_to_list('Variableset/constantlist'+dataset+'.csv')
    parse_data,log_sentence=GeLT.log_to_model(log_content,stage='parse',regx=[],regx_use=False,dataset=dataset,)
    train_data,_=GeLT.log_to_model(log_content,stage='train',regx=setting['regex'],regx_use=False,
                                        dataset=dataset,variablelist=variablelist,constantlist=constantlist,use_optim=args.use_optimized)

    # log_groups, PA_of_batches,constant,variable,constant1,variable1,constant2,variable2,PA_of_batches_withoutTTT,PA_of_batches_origin
    (log_groups, log_groups_without_TTT,log_groups_origin,
     PA_of_batches,PA_of_batches_withoutTTT,PA_of_batches_origin,
     constant,variable,constant1,variable1,constant2,variable2
    ) = GeLT.online_parsing_withTTT(train_data=train_data,parse_data=parse_data,log_sentence=log_sentence,
                                     threshold=setting['threshold'], ground_truth_list=ground_truth_list,
                                     modelpath=f'torch_model_100_5/model_{dataset}_2k', model=model, model2=model2,
                                     model3=model3, dataset=dataset, do_train=True, use_optim=args.use_optimized)

    predictions = list(range(len(log_sentence)))
    for key,value in log_groups.items():
        for idx in value:
            predictions[idx] = key
    for idx, pred in enumerate(predictions):
        if pred == idx:
            print("ERROR")


    def plg(lg):
        print("{\n" + "\n".join("{!r}: {!r},".format(k, v) for k, v in lg.items()) + "\n}")
        pass
    # plg(log_groups)
    # print(predictions[0])

    if args.use_optimized:
        outdir = Path(f'output/GeLT'.format(data_type))
    else:
        outdir = Path(f'output/Log3T'.format(data_type))

    os.makedirs(outdir, exist_ok=True)
    _df = pd.DataFrame(data=list(zip(log_content,predictions)))
    _df.to_csv(outdir / "{}_{}.log_structured.csv".format(dataset,data_type),
               header=['Content','EventTemplate'], index=False)

    predictions = list(range(len(log_sentence)))
    for key,value in log_groups_origin.items():
        for idx in value:
            predictions[idx] = key

    outdir = Path(f'output/origin'.format(data_type))
    os.makedirs(outdir, exist_ok=True)
    _df = pd.DataFrame(data=list(zip(log_content,predictions)))
    _df.to_csv(outdir / "{}_{}.log_structured.csv".format(dataset,data_type),
               header=['Content','EventTemplate'], index=False)

    end_time=datetime.datetime.now()
    print('Process time = '+str(end_time-start))
