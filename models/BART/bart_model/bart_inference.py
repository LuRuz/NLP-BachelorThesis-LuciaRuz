# -*- coding: utf-8 -*-
import sys

import gcc 
import datasets
import pandas as pd 
from fastai.text.all import *
from transformers import *

from blurr.data.all import *
from blurr.modeling.all import *

import os
import time


BUCKET_NAME='transformers-lucia'

remove_dataset = False

#List with the datasets
datasetInfo = [['cnn_dailymail', '3.0.0', 'article', 'highlights'],
               ['gigaword', '1.2.0', 'document', 'summary'],
               ['xsum', '1.1.0', 'document', 'summary'],
               ['biomrc', 'biomrc_large_B', 'abstract','title']]

source_dataset = sys.argv[1]
numDataset = int(sys.argv[2])
w_wt_sum = int(sys.argv[3])

nameDataset = datasetInfo[numDataset][0]

path_model='./bart_model/models/'+nameDataset+'.pkl'
print('Loading the model from ',path_model)

if not os.path.isfile(path_model):
    #we load it download it from the bucket
    print('we download the model from ',BUCKET_NAME)
    gcc.download_blob(BUCKET_NAME,'tfg/models/BART/'+nameDataset+'.pkl',path_model)

# model = load_learner(fname=path_model)
model = load_learner(fname=path_model)
print('{} model was loaded'.format(path_model))


#We use the model over the texts from the test dataset
start_time = time.time()

# Download datasets
if ( source_dataset == "transformers"):
    nameDataset=datasetInfo[numDataset][0]      #name of the dataset
    versionDataset=datasetInfo[numDataset][1]   #version of the dataset
    text_field=datasetInfo[numDataset][2]       #name of the field that contains the input texts
    summary_field=datasetInfo[numDataset][3]    #name of the field that contains the summaries

    print("We will perform prediction for the dataset", nameDataset)


    print('This may take some minutes...')
    test_data = datasets.load_dataset(nameDataset, versionDataset, split='test')
    print('size of the test dataset {} = {}'.format(nameDataset,len(test_data)))

    #we load it into a pandas dataframe
    df_test = pd.DataFrame(test_data)
    
else: 
    path_dataset = "./bart_model/datasets/"+source_dataset+".csv"
    #Must be allocated on notebook or on the bucket
    if not os.path.exists(path_dataset):
        remove_dataset = True
        print('we download the model from ',BUCKET_NAME)
        #download_blob(bucket_name, source_blob_name, destination_file_name)
        gcc.download_blob(BUCKET_NAME,'tfg/datasets/'+source_dataset+'.csv',path_dataset)
        
    #Read CSV with data to summarize
    test_data = pd.read_csv(path_dataset, encoding='utf-8', delimiter='|')
    text_field = 'Text'
    summary_field = 'gold_summary'
    df_test = pd.DataFrame(test_data)
    

input_texts = []        #list to save the input texts
gold_summaries = []     #list to save the summaries from the test dataset
predicted_summaries = []    #list to save the summaries created by the model

total = len(test_data)

if (w_wt_sum == 1 or w_wt_sum == -1):
    #This for traverses all texts from the test dataet
    for index, row in df_test.iterrows():
        #save the input text
        input_texts.append(row[text_field])
        #save its corresponding summary
        gold_summaries.append(row[summary_field])

        print("Predicting summary for {} / {} ".format(index,total))

        #now we use the model to generate the summary
        predicted = model.blurr_generate(row[text_field])
        #we save the generated summary
        predicted_summaries.append(predicted[0])

        #print("--- time ---" , ((time.time() - start_time)/60)," min --- ", (time.time() - start_time),' sec ---') 


    #Now, we create a dataframe with these three lists and save it into a csv
    data={"input_text": input_texts, "gold_summary": gold_summaries, "predicted_summary" : predicted_summaries}
    #data = {"newSummaries":predicted_summaries,'originalSummaries':gold_summaries,'fullTexts':input_text}

    #we create a dataframe to save the input
    df = pd.DataFrame(data, columns = ['input_text', 'gold_summary', 'predicted_summary'])
    #df = pd.DataFrame( data , columns = ["newSummaries","originalSummaries","fullTexts"])
    
elif (w_wt_sum == 0):
     #This for traverses all texts from the test dataet
    for index, row in df_test.iterrows():
        #save the input text
        input_texts.append(row[text_field])

        print("Predicting summary for {} / {} ".format(index,total))

        #now we use the model to generate the summary
        predicted = model.blurr_generate(row[text_field])
        #we save the generated summary
        predicted_summaries.append(predicted[0])

        #print("--- time ---" , ((time.time() - start_time)/60)," min --- ", (time.time() - start_time),' sec ---') 


    #Now, we create a dataframe with these three lists and save it into a csv
    data={"input_text": input_texts, "predicted_summary" : predicted_summaries}

    #we create a dataframe to save the input
    df = pd.DataFrame(data, columns = ['input_text', 'predicted_summary'])

#The csv for saving the predictions should be called as the dataset name :
path_predictions='./bart_model/summaries/'+nameDataset+'.csv' #name of the file to save them
df.to_csv(path_predictions,encoding='utf-8', delimiter='|') #we save the dataframe into the csv
print('generated summaries were saved into {}'.format(path_predictions))

print('uploading to the bucket')
# gcc.upload_blob(BUCKET_NAME,path_predictions,'tfg/summaries/BART/'+nameDataset+'.csv' )
# os.remove(path_predictions)

if remove_dataset:
    print('now we remove the file with the generated summaries from the instance...',path_predictions)
    os.remove(path_dataset)

print('That is all!!!')