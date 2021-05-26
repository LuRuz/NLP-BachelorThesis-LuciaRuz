# -*- coding: utf-8 -*-
#!pip install gcloud
#!pip install git+https://github.com/dmmiller612/bert-extractive-summarizer.git@small-updates
#!pip install rouge

import pandas as pd
import numpy as np

import gcc
import os


import datasets
import sys
from summarizer import Summarizer

BUCKET_NAME='transformers-lucia'

datasetInfo = [['cnn_dailymail', '3.0.0', 'article', 'highlights'],
               ['gigaword', '1.2.0', 'document', 'summary'],
               ['xsum', '1.1.0', 'document', 'summary'],
               ['biomrc', 'biomrc_large_B', 'abstract','answer']]

source_dataset = sys.argv[1]
dataset_number = int(sys.argv[2])
w_wt_sum = int(sys.argv[2])
model_name = sys.argv[3]
remove_dataset = False


"""MAIN FUNCTION"""
#General model
model = Summarizer()

#Save the relevant data
name_dataset = datasetInfo[dataset_number][0]            
version_dataset = datasetInfo[dataset_number][1]
text_column = datasetInfo[dataset_number][2]
summary_column = datasetInfo[dataset_number][3]

print(name_dataset, version_dataset)

if (source_dataset == "transformers"):
    #load dataset to summarize
    test_data = datasets.load_dataset(name_dataset, version_dataset, split="test")
else:
    print(source_dataset)
    print()
    path_dataset = "./datasets/"+source_dataset+".csv"
    
    #Must be allocated on notebook or on the bucket
    if not os.path.exists(path_dataset):
        remove_dataset = True
        print('we download the model from ',BUCKET_NAME)
        #download_blob(bucket_name, source_blob_name, destination_file_name)
        gcc.download_blob(BUCKET_NAME,'tfg/datasets/'+source_dataset+'.csv',path_dataset)
        
    #Read CSV with data to summarize
    test_data = pd.read_csv(path_dataset, encoding='utf-8', delimiter='|')
    text_column = 'Text'
    summary_column = 'gold_summary'
    
##Do all predictions
newSummary = []
originalText = []
totalText = len(test_data[text_column])

## Path to save predictions
path_results = './summaries/' + model_name + '-' + name_dataset + '.csv'

##Download model
if (model_name == 'albert'):
    from transformers import AlbertTokenizer, AlbertModel
    albert_model = AlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True)
    albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    model = Summarizer(custom_model=albert_model, custom_tokenizer=albert_tokenizer, random_state = 7)

elif(model_name == 'distilbert'):
    from transformers import DistilBertModel, DistilBertTokenizer
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = Summarizer(custom_model=distilbert_model, custom_tokenizer=distilbert_tokenizer, random_state = 7)
    
elif(model_name == 'T5'):
    from transformers import AutoModelWithLMHead, AutoTokenizer
    model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-summarize-news',output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-summarize-news')

    

#It has source summary
if (w_wt_sum == 1 or source_dataset == 'transformers'):
    originalSummary = []

    if (model_name == 'albert' or model_name == 'distilbert'):
        for numText, fullText in enumerate(test_data[text_column]):
            #Clean full text and make prediction
            summary_generated = model(fullText)

            #Save full text and the prediction
            originalText.append(fullText)
            newSummary.append(summary_generated)

            #Save original summary
            originalSum = test_data[summary_column][numText]
            originalSummary.append(originalSum)

            print("Analizado ", numText," de ", totalText)
    elif (model_name == 'T5'):
        max_length = 8192
        
        #summarizer loop
        for numText, fullText in enumerate(test_data[text_column]):   
            # generate ids
            input_ids = tokenizer.encode(fullText, return_tensors="pt", add_special_tokens=True)
            generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
            
            #Clean full text and make prediction
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]            
            summary_generated = preds[0]

            #Save full text and the prediction
            originalText.append(fullText)
            newSummary.append(summary_generated)

            #Save original summary
            originalSum = test_data[summary_column][numText]
            originalSummary.append(originalSum)

            print("Analizado ", numText," de ", totalText)
    
            
    #Save results
    testExport = {"input_text" : originalText, "gold_summary": originalSummary, "predicted_summary": newSummary }
    exportData = pd.DataFrame(testExport, columns = ["input_text","gold_summary","predicted_summary"])
    exportData.to_csv(path_results, index = False, sep='|')
        
#Inference without source_summary
elif (w_wt_sum == 0 and not source_dataset == 'transformers'):
    for numText, fullText in enumerate(test_data[text_column]):
        #Clean full text and make prediction
        summary_generated = summarize(fullText)

        #Save full text and the prediction
        originalText.append(fullText)
        newSummary.append(summary_generated)

        print("Analizado ", numText," de ", totalText)

    #Save results
    testExport = {"input_text" : originalText, "predicted_summary": newSummary }
    exportData = pd.DataFrame(testExport, columns = ["input_text","predicted_summary"])
    exportData.to_csv(path_results, index = False, sep='|')

#upload to bucket and remove from notebook
gcc.upload_blob(BUCKET_NAME,path_results,'tfg/summaries/HuggingFace/'+ model_name+'-'+name_dataset+'.csv')
os.remove(path_results)

if remove_dataset:
    os.remove(path_dataset)

print("SUCCESSFUL")