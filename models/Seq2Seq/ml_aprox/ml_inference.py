#python3 ml_inference.py arg1 where arg1 is the dataset -> transformers or 'others'
#if the dataset is 'others' it must exist a folder in the bucket with the name introduced

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import sys
import os
from datasets import load_dataset, load_metric
import csv
import gcc

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict

import datasets
import pandas as pd


BUCKET_NAME = "transformers-lucia"
path_predictions = "./ml_aprox/summaries/"


def main():
    
    # first of all, lets parse the arguments from command-line
    source_dataset = sys.argv[1]
    dataset_number = int(sys.argv[2])
        
    if(source_dataset == 'transformers'):
        #summarization with transformers datasets
        #List with the datasets
        datasetInfo = [['cnn_dailymail', '3.0.0', 'article', 'highlights'],
               ['gigaword', '1.2.0', 'document', 'summary'],
               ['xsum', '1.1.0', 'document', 'summary'],
               ['biomrc', 'biomrc_large_B', 'abstract','answer']]
        
        if(dataset_number == -1):
            #execute the code for all the datasets
            for datasetToSummarize in datasetInfo:

                #exctract dataset info
                name_dataset=datasetToSummarize[0]
                print("Predicting in " + name_dataset + "dataset.")
                
                version_dataset=datasetToSummarize[1]
                text_column=datasetToSummarize[2]
                summary_column=datasetToSummarize[3]

                #load datasets
                test_dataset = load_dataset(name_dataset, version_dataset, split="test")

                #create a list with input text, source summary and new summary
                full_summarization=[["input_text","gold_summary","predicted_summary"]]
                length = len(test_dataset)

                for num_row, full_row in enumerate(test_dataset):
                    content = test_dataset[num_row][text_column]
                    source_summary = test_dataset[num_row][summary_column]

                    #Generate new summary
                    summary = procesing(content)
                    full_summarization.append([content,source_summary,summary])
                    print(str(num_row) + " of " + str(length) + " completed...")
                
                #Save all summaries in a new csv
                save_to_csv(name_dataset+".csv", full_summarization)

                #Save it on the bucket and remove from notebook
                gcc.upload_blob(BUCKET_NAME,path_predictions+name_dataset+'.csv','tfg/summaries/machine-learning/'+name_dataset+'.csv' )
                print('now we remove the file with the generated summaries from the instance...',path_predictions)
                os.remove(path_predictions+name_dataset+'.csv')
        
        elif (dataset_number < 4 and dataset_number > -1):
            #Save the relevant data
            name_dataset = datasetInfo[dataset_number][0]
            print("Procesing "+name_dataset+" dataset")
            
            version_dataset = datasetInfo[dataset_number][1]
            text_column = datasetInfo[dataset_number][2]
            summary_column = datasetInfo[dataset_number][3]
            
            print(name_dataset, version_dataset)
            #load dataset to summarize
            test_dataset = load_dataset(name_dataset, version_dataset, split = "test")
            
            #create a list with input text, source summary and new summary
            full_summarization=[["input_text","gold_summary","predicted_summary"]]
            length = len(test_dataset)
            
            for num_row, full_row in enumerate(test_dataset):
                content = test_dataset[num_row][text_column]
                source_summary = test_dataset[num_row][summary_column]
                
                #Generate new summary
                summary = procesing(content)
                full_summarization.append([content,source_summary,summary])
                print(str(num_row) + " of " + str(length) + " completed...")
                
                
            #Save all summaries in a new csv
            save_to_csv(name_dataset+".csv", full_summarization)
            
            #Save it on the bucket and remove from notebook
            gcc.upload_blob(BUCKET_NAME,path_predictions+name_dataset+'.csv','tfg/summaries/machine-learning/'+name_dataset+'.csv' )
            print('now we remove the file with the generated summaries from the instance...',path_predictions)
            os.remove(path_predictions+name_dataset+'.csv')
                
        else:
            print("THE DATASET DOES NOT EXIST")
                
                
        
    else:
        if (dataset_number != 0 and dataset_number != 1):
            print("THIS OPTION IS NOT VALID, TRY AGAIN")
        
        else:
            #The dataset must be a .csv
            path_dataset = "./ml_aprox/datasets/"+source_dataset+".csv"

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
            input_text = pd.read_csv(path_dataset, encoding='utf-8', delimiter='|', usecols=['Text'])
            length = len(input_text['Text'])

            #If has gold_summary...
            if (dataset_number == 1 ):
                gold_summary = pd.read_csv(path_dataset, encoding='utf-8', delimiter='|', usecols=['gold_summary'])
                full_summarization=[["input_text","gold_summary","predicted_summary"]]
                
                for num, content in enumerate(input_text['Text']):                
                    #Generate new summary
                    summary = procesing(content)

                    #Save all in a list
                    source_summary = gold_summary['gold_summary'][num]
                    full_summarization.append([content,source_summary,summary])

                    print(str(num+1) + " of " + str(length) + " completed...")
            
            # If hasnt summary of reference...
            else:
                full_summarization=[["input_text","predicted_summary"]]
                
                for num, content in enumerate(input_text['Text']):                
                    #Generate new summary
                    summary = procesing(content)

                    #Save all in a list
                    full_summarization.append([content,summary])

                    print(str(num+1) + " of " + str(length) + " completed...")
                    
            #Save all summaries in a new csv
            save_to_csv(source_dataset+".csv", full_summarization)
            
            #Save it on the bucket and remove from notebook
#             gcc.upload_blob(BUCKET_NAME,path_predictions+source_dataset+'.csv','tfg/summaries/machine-learning/'+source_dataset+'.csv' )
#             print('now we remove the file with the generated summaries from the instance...',path_predictions)
            
#             os.remove(path_predictions+source_dataset+'.csv')
            if remove_dataset:
                os.remove(path_dataset)
        

    
    
def procesing(content):
    # remove stop words and punctuation
    content = sanitize_input(content)

    # tokenize sentences and words
    sentence_tokens, word_tokens = tokenize_content(content)

    # rank sentences based on their content
    sentence_ranks = score_tokens(word_tokens, sentence_tokens)

    # summarize the text and return it    
    return summarize(sentence_ranks, sentence_tokens)

def read_file(path):
    try:
        with open(path, 'r') as file:
            return file.read()
    except IOError as e:
        print("Fatal Error: File ({}) could not be located or is not readable\n".format(path), e)

def sanitize_input(data):
    replace = {
        ord('\f') : ' ',
        ord('\t') : ' ',
        ord('\n') : ' ',
        ord('\r') : None
    }
    return data.translate(replace)

def tokenize_content(content):
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(content.lower())

    return [
        sent_tokenize(content),
        [word for word in words if word not in stop_words]
    ]

def score_tokens(filtered_words, sentence_tokens):
    word_freq = FreqDist(filtered_words)

    ranking = defaultdict(int)

    for i, sentence in enumerate(sentence_tokens):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                ranking[i] += word_freq[word]
    
    return ranking

def summarize(ranks, sentences):    
    indexes = nlargest(1, ranks, key=ranks.get)
    final_sentences = [sentences[j] for j in sorted(indexes)]
    return ' '.join(final_sentences)

def save_to_csv(csv_name, results_list):
    with open(path_predictions+csv_name,'w') as f:
        write = csv.writer(f,delimiter='|')
        write.writerows(results_list)
        
    print("CSV saved correct")


if __name__ == "__main__":
    main()