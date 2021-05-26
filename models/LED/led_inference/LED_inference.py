#python3 LED_inference.py arg1 where arg1 is the dataset -> transformers or 'others'
#if the dataset is 'others' it must exist a csv on 'datasets' folder with the name introduced

from datasets import Dataset
import pandas as pd
import torch

import os
import csv
import sys
import gcc

from transformers import LEDForConditionalGeneration, LEDTokenizer
from datasets import load_dataset, load_metric


BUCKET_NAME = 'transformers-lucia'

#dataset to summarize
source_dataset=sys.argv[1]
num_dataset=int(sys.argv[2])

#Dowload model pre-trained
print("Downloading pre-trained model...")
tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv").to("cuda").half()

#Function to generate predictions
def generate_answer(batch):
  inputs_dict = tokenizer(batch['Text'], padding="max_length", max_length=16384, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")

  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, max_length=512, num_beams=4)
  batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  return batch

#Function to generate csv with results
def generate_csv(source_summary, results, csv_name):
    #Create csv with results
    createCSV=[["input_text","gold_summary","predicted_summary"]]

    for number, element in enumerate(results):
        print(element['Text'])
        print()
        print(source_summary[number])
        print()
        print(element['predicted_abstract'])
        createCSV.append([element['Text'],source_summary[number],element['predicted_abstract']])

    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file,delimiter='|')
        writer.writerows(createCSV)



#To evaluate with transformers datasets
if(source_dataset=='transformers'):
    #List with the datasets
    datasetInfo = [['cnn_dailymail', '3.0.0', 'article', 'highlights'],
                   ['gigaword', '1.2.0', 'document', 'summary'],
                   ['xsum', '1.1.0', 'document', 'summary'],
                   ['biomrc', 'biomrc_large_B', 'abstract','title']]
    
    
    #Execute all the transformer's datasets
    if (num_dataset == -1):
        print("Executing all transformer's datasets")
        
        #execute the code for all the datasets
        for datasetToSummarize in datasetInfo:
            name_dataset=datasetToSummarize[0]
            version_dataset=datasetToSummarize[1]
            test_dataset_full = load_dataset(name_dataset, version_dataset, split="test")

            print("Dataset selected: "+name_dataset)
            
            #Create a dataframe with column name text to process after
            text_column_name = datasetToSummarize[2]
            text_column = test_dataset_full[text_column_name]
            text_column = pd.DataFrame(text_column, columns=["Text"])
            test_dataset = Dataset.from_pandas(text_column)

            summary_column = datasetToSummarize[3]

            #Generate results
            results = test_dataset.map(generate_answer, batched=True, batch_size=2)

            #Save results
            path_results = "./led_inference/summaries/"+name_dataset+".csv"
            generate_csv(test_dataset_full[summary_column], results, path_results)

            #Move results to bucket and remove from notebook
            gcc.upload_blob(BUCKET_NAME,path_results,'tfg/summaries/LED/'+name_dataset+'.csv' )
            os.remove(path_results)
        
        #Save results
        generate_csv(results, "summaries/"+name_dataset+".csv")
    
    #Execute specific huggingface dataset
    elif (num_dataset < 4 and num_dataset > -1):
        name_dataset = datasetInfo[num_dataset][0]
        print("Dataset selected: "+name_dataset)
        
        version_dataset = datasetInfo[num_dataset][1]
        test_dataset_full = load_dataset(name_dataset, version_dataset, split="test")
        
        #Create a dataframe with column name text to process after
        text_column_name = datasetInfo[num_dataset][2]
        text_column = test_dataset_full[text_column_name]
        text_column = pd.DataFrame(text_column, columns=["Text"])
        test_dataset = Dataset.from_pandas(text_column)
        
        summary_column = datasetInfo[num_dataset][3]

        #Generate results
        results = test_dataset.map(generate_answer, batched=True, batch_size=2)
        
        #Save results
        path_results = "./led_inference/summaries/"+name_dataset+".csv"
        generate_csv(test_dataset_full[summary_column], results, path_results)
        
        #Move results to bucket and remove from notebook
        gcc.upload_blob(BUCKET_NAME,path_results,'tfg/summaries/LED/'+name_dataset+'.csv' )
        os.remove(path_results)
        
    else:
        print("THE DATASET DOES NOT EXIST")
        
        
else:
    #The dataset must be a .csv
    path_dataset = "./led_inference/datasets/"+source_dataset+".csv"
     
    #Must be allocated on notebook or on the bucket
    if not os.path.exists(path_dataset):
        remove_dataset = True
        print('we download the model from ',BUCKET_NAME)
        #download_blob(bucket_name, source_blob_name, destination_file_name)
        gcc.download_blob(BUCKET_NAME,'tfg/datasets/'+source_dataset+'.csv',path_dataset)
        
    else:
        remove_dataset = False
        print("Dataset is on notebook")

    
    #Read CSV with data to summarize
    df = pd.read_csv(path_dataset, encoding='utf-8', delimiter='|', usecols=['Text'])
    print(df.keys())
    test_dataset = Dataset.from_pandas(df)
    
    #Generate results
    results = test_dataset.map(generate_answer, batched=True, batch_size=2)
        
    #Save results
    path_results = "./led_inference/summaries/"+source_dataset+"_summaries.csv"
    
    #Dataset without source summary
    if (num_dataset == 0):
        createCSV=[["input_text","predicted_summary"]]

        for element in results:
            createCSV.append([element['Text'],element['predicted_abstract']])
        
        print("Creating .csv with results...")
        with open(path_results, 'w', newline='') as file:
            writer = csv.writer(file,delimiter='|')
            writer.writerows(createCSV)
        
    #Dataset with source summary named 'gold_summary'
    elif (num_dataset == 1):
        summaries_column = pd.read_csv(path_dataset, encoding='utf-8', delimiter='|', usecols=['gold_summary'])
        generate_csv(summaries_column['gold_summary'], results, path_results)
        
    #Move results to bucket and remove from notebook
    gcc.upload_blob(BUCKET_NAME,path_results,'tfg/summaries/LED/'+source_dataset+'_summaries.csv' )
    
    #Remove irrelevant data
    os.remove(path_results)
    
    if remove_dataset:
        os.remove(path_dataset)
        
