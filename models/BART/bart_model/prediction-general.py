# -*- coding: utf-8 -*-

#    !pip install ohmeow-blurr -q
#    !pip install nlp -q

import nlp
from fastai.text.all import *
from transformers import *

from blurr.data.all import *
from blurr.modeling.all import *

import pandas as pd 
import gcc 
import os
import time

import sys

root='./'

BUCKET_origin='textsummarization-sepln'
BUCKET_destination='transformers-lucia'
dataset_name=sys.argv[1]

#Set to False if we want to keep the file with the predictions after uploading it to the bucket
deletePredictions=False  



#Please, select the dataset 
nameDataset=sys.argv[2]      #name of the field that contains the input texts

print("We will perform prediction for the dataset", nameDataset)


#print('This may take some minutes...')
# test_data = nlp.load_dataset(nameDataset, versionDataset, split='test')
# print('size of the test dataset {} = {}'.format(nameDataset,len(test_data)) )

data_dir='./datasets/'+dataset_name+'/'
test_data = os.listdir(data_dir)
total = len(test_data)
print('size of the test dataset = '+str(total))

# we load it into a pandas dataframe
# df_test = pd.DataFrame(test_data)

path_model=root+'models/'+nameDataset+'.pkl'
print('Loading the model from ',path_model)

if not os.path.isfile(path_model):
    #we load it download it from the bucket
    print('we download the model from ',BUCKET_origin)
    gcc.download_blob(BUCKET_origin,nameDataset+'.pkl',path_model)

model = load_learner(fname=path_model)
print('{} model was loaded'.format(path_model))


#We use the model over the texts from the test dataset
start_time = time.time()

input_texts = []        #list to save the input texts
# gold_summaries = []     #list to save the summaries from the test dataset
predicted_summaries = []    #list to save the summaries created by the model


# #This for traverses all texts from the test dataet
for index, row in enumerate(test_data):
    with open(data_dir+row) as doc:
        full_text = doc.read()
        input_texts.append(full_text)
        
        print("Predicting "+str(index)+" of "+str(total)+" ...")
        predicted = model.blurr_summarize(full_text)
        
        predicted_summaries.append(predicted[0])
        
        print("--- time ---" , ((time.time() - start_time)/60)," min --- ", (time.time() - start_time),' sec ---')
        
#Now, we create a dataframe with these three lists and save it into a csv
data={"input_text": input_texts, "predicted_summary" : predicted_summaries}
# data = {"newSummaries":predicted_summaries,'originalSummaries':gold_summaries,'fullTexts':input_text}
    
# #we create a dataframe to save the input
df = pd.DataFrame(data, columns = ['input_text', 'predicted_summary'])

#The csv for saving the predictions should be called as the dataset name :
path_predictions=root+'outputs/'+dataset_name+'.csv' #name of the file to save them
df.to_csv(path_predictions,encoding='utf-8') #we save the dataframe into the csv
print('generated summaries were saved into {}'.format(path_predictions))

print('uploading to the bucket')
gcc.upload_blob(BUCKET_destination,path_predictions,'toSummarize/Summaries/'+nameDataset+'_'+dataset_name+'.csv' )


if deletePredictions:
    print('now we remove the file with the generated summaries from the instance...',path_predictions)
    os.remove(path_predictions)

print('That is all!!!')