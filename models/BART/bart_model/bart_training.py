# -*- coding: utf-8 -*-
import sys

import gcc
import datasets
import pandas as pd
from fastai.text.all import *
from transformers import *

from blurr.data.all import *
from blurr.modeling.all import *

#List with the possible datasets

datasetInfo = [['cnn_dailymail', '3.0.0', 'article', 'highlights', 'facebook/bart-large-cnn'],
               ['gigaword', '1.2.0', 'document', 'summary','facebook/bart-large'],
               ['xsum', '1.1.0', 'document', 'summary','facebook/bart-large-xsum'],
               ['biomrc', 'biomrc_large_B', 'abstract','answer','facebook/bart-large']]

#Please, select the index of the dataset
numDataset = int(sys.argv[1]) 

name_dataset=datasetInfo[numDataset][0]     #name of the dataset
version_dataset=datasetInfo[numDataset][1]  #version of the dataset
text_field=datasetInfo[numDataset][2]       #name of the field containing the input texts
summary_field=datasetInfo[numDataset][3]    #name of the field containing the summaries
pretrained_model_name=datasetInfo[numDataset][4] #name of the pretrained model at huggingface


BUCKET_NAME='transformers-lucia'


print('dataset selected:')
print(name_dataset,version_dataset,text_field,summary_field)


#we load the dataset
print('loading {} {}...'.format(name_dataset,version_dataset))
raw_data = datasets.load_dataset(name_dataset, version_dataset)

print('Number of instances in  {} {}'.format(name_dataset,version_dataset))
print('#instances in the training dataset:',len(raw_data['train']))
print('#instances in the  validation dataset:',len(raw_data['validation']))
print('#instances in the test dataset:',len(raw_data['test']))

"""We won't use the validation dataset to fix the hyperparameters, so we can use it to train the model:"""
df_train = pd.DataFrame(raw_data['train'])
#we append the validation data to the training data to obtain a larger training dataset
df_train=df_train.append(pd.DataFrame(raw_data['validation']))
print('#instances in the extended training dataset:',len(df_train))

print('Now, we will define the BART architecture...')
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR_MODEL_HELPER.get_hf_objects(pretrained_model_name, 
                                                                               model_cls=BartForConditionalGeneration)

hf_arch, type(hf_config), type(hf_tokenizer), type(hf_model)

#building the datablock to store the data

print('\tcreating the datablock...')

text_gen_kwargs = default_text_gen_kwargs(hf_config, hf_model, task='summarization');

hf_batch_tfm = HF_Seq2SeqBeforeBatchTransform(hf_arch, hf_config, hf_tokenizer, hf_model, max_length=256, max_tgt_length=130, text_gen_kwargs=text_gen_kwargs)

blocks = (HF_Seq2SeqBlock(before_batch_tfm=hf_batch_tfm), noop)

dblock = DataBlock(blocks=blocks, get_x=ColReader(text_field),get_y=ColReader(summary_field), splitter=RandomSplitter())


print('\tdatablock was defined...')
#We load the training dataset into the datablock
dls = dblock.dataloaders(df_train, bs=2)
print('\tdatablock was loaded...',len(dls.train.items),len(dls.valid.items))


#It's always a good idea to check out a batch of data and make sure the shapes look right."""

b = dls.one_batch()
print('Checking out a batch: ', len(b), b[0]['input_ids'].shape, b[1].shape)

#Even better, we can take advantage of blurr's TypeDispatched version of `show_batch` to look at things a bit more intuitively."""

print("Show two instances:")
dls.show_batch(dataloaders=dls, max_n=2)

"""## Training"""
print("First, we define our own metric and the model to train")

#Metric definition
seq2seq_metrics = {
        'rouge': {
            'compute_kwargs': { 'rouge_types': ["rouge1", "rouge2", "rougeL"], 'use_stemmer': True },
            'returns': ["rouge1", "rouge2", "rougeL"]
        },
        'bertscore': {
            'compute_kwargs': { 'lang': 'en' },
            'returns': ["precision", "recall", "f1"]
        }
    }

#Model
model = HF_BaseModelWrapper(hf_model)
learn_cbs = [HF_BaseModelCallback]
fit_cbs = [HF_Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

learn = Learner(dls, 
                model,
                opt_func=ranger,
                loss_func=CrossEntropyLossFlat(),
                cbs=learn_cbs,
                splitter=partial(seq2seq_splitter, arch=hf_arch)).to_fp16()

learn.create_opt() 
learn.freeze()

print('model was defined!')

print('Training the model...')
learn.fit_one_cycle(1, lr_max=3e-5, cbs=fit_cbs)

print('\tModel was trained')

print('We will now save the model')

path_model='./bart_model/models/'+name_dataset+'.pkl'

#Save the model
learn.metrics = None
learn.export(fname=path_model)
print('model was saved into {}'.format(path_model))


import gcc
print('We also move the model to the bucket')
gcc.upload_blob(BUCKET_NAME,path_model,'tfg/models/BART/'+name_dataset+'.pkl' )

  
print('That is all!!!')