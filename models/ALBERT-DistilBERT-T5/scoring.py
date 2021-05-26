# -*- coding: utf-8 -*-
import sys

import gcc #functions to upload and download from the bucket
import pandas as pd
import os

from rouge import Rouge #alows to compute **rouge-1**, **rouge-2** and **rouge-l** 

#pip install rouge-metric
# from rouge_metric import PerlRouge #rouge1, rouge2, rouge3, rougeL, rougeW, rougeS, rougeSU

import nltk   #for BLEU


extendedRouge=False #Set to True if you want to calculate all rouge scores

#Set to False if we want to keep the file with the predictions after uploading it to the bucket
deletePredictions=False 

# Select the dataset
datasetInfo = [['cnn_dailymail'], ['gigaword'], ['xsum'], ['biomrc']]

#Please, select the dataset 
numDataset = int(sys.argv[1]) 
nameModel = sys.argv[2]

nameDataset=datasetInfo[numDataset][0]

print("Evaluating ", nameDataset)

BUCKET_NAME='transformers-lucia'  #name of the bucket where we saved the generated summaries



#loading the predictions
path_predictions='./summaries/'+nameModel+'-'+nameDataset+'.csv'
if not os.path.exists(path_predictions):
    remove_predictions = True
    print('{} does not exist!!!'.format(path_predictions))
    print('we download the model from ',BUCKET_NAME)
    #download_blob(bucket_name, source_blob_name, destination_file_name)
    gcc.download_blob(BUCKET_NAME,'tfg/summaries/BERT/'+nameModel+'-'+nameDataset+'.csv',path_predictions)
else:
    remove_predictions = False
    print("Model is on notebook")
    
#we load the csv file
df=pd.read_csv(path_predictions, usecols=["input_text","gold_summary","predicted_summary"], engine="python", encoding='utf-8',error_bad_lines=False, delimiter='|')


predicted_summaries=df['predicted_summary'].tolist() #list with the generated summaries
gold_summaries=df['gold_summary'].tolist() #list with the original summaries

#Remove rows with empty cells and summaries with errors
list_to_modify = []
rouge=Rouge()

##Detect rows
for num, i in enumerate(predicted_summaries):
    if not isinstance(i,str):
        list_to_modify.append(num)
    else:
        predict_aux=i
        gold_aux=gold_summaries[num]
        try:
            scores = rouge.get_scores(predict_aux, gold_aux, avg=True)
        except:
            list_to_modify.append(num)

##delete rows
for minus, number in enumerate(list_to_modify):
    del predicted_summaries[number-minus]
    del gold_summaries[number-minus]
 

print('{} summaries were loaded'.format(len(gold_summaries)))

#We are going to use **rouge-1**, **rouge-2** and **rouge-l** to evaluate with test dataframe. 
#To do this, we will use the library **rouge** (https://pypi.org/project/rouge/).
#pip install rouge


print('Evaluation for ', nameDataset)

print('calculating (basic) Rouge...')

rouge = Rouge()
scores = rouge.get_scores(predicted_summaries, gold_summaries, avg=True)

print('Scores obtained by using the Rouge library:\n',scores)

#we save into the instance

path_scores='./evaluation/'+nameModel+'-'+nameDataset+'.txt'
f = open(path_scores, 'w')
f.write("Scores for {}\n".format(nameDataset))
f.write('____________________________________\n')
for metric in scores: #metric will be rouge-1, rouge-2 or rouge-L
    f.write("{}\n".format(metric))
    f.write("\n{}\n".format(scores[metric]))
f.write('\n\n')
f.close()
print('Basic ROUGE scores were added to the file {}'.format(path_scores))

#### rouge1, rouge2, rouge3, rougeL, rougeW, rougeS, rougeSU
# https://pypi.org/project/rouge-metric/

if extendedRouge:
    print('calculating extended Rouge...')

    #The parameters allow us to indicate what Rouge metrics we want to obtain:
    rouge = PerlRouge(rouge_n_max=3, rouge_l=True, rouge_w=True,
        rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)


    #then, we use the 
    scores = rouge.evaluate(predicted_summaries, gold_summaries)
    print(scores)

    """We also save them into the file with the scores:"""

    f = open(path_scores, 'a')
    f.write("Extended ROUGE Scores for {}\n".format(nameDataset))
    f.write('____________________________________\n')
    for metric in scores: #metric will be rouge-1, rouge-2 or rouge-L
        f.write("{}\n".format(metric))
        f.write("\n{}\n".format(scores[metric]))
    f.write('\n\n')
    f.close()

    print('Extended ROUGE scores were added to the file {}'.format(path_scores))


### BLEU"""

print('calculating BLEU...')
total = 0
for i in range(len(gold_summaries)):
    total += nltk.translate.bleu_score.sentence_bleu([gold_summaries[i]], predicted_summaries[i],weights=(1.0, 0, 0, 0))

BLEUscore = total / len(gold_summaries)
print('BLEU:',BLEUscore)

#add the score to the score file
f = open(path_scores, 'a')
f.write("BLEU score for {}:{}\n".format(nameDataset,BLEUscore))
f.close()
print('BLEU score was added to the file {}'.format(path_scores))


#Finally, we will save the file with all the score into the GCC bucket
print('uploading to the bucket:',BUCKET_NAME)
gcc.upload_blob(BUCKET_NAME,path_scores,'tfg/scores/BERT/'+nameModel+'-'+nameDataset+'-score.txt' )


if remove_predictions:
    os.remove(path_predictions)
    
    
print('That is all!!!')