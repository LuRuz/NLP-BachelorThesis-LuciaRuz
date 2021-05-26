#!/bin/bash

pip install -r requirements.txt

# To call this funcition ./summarizer.sh param1 where param1 is the name of the dataset to summarize
source_dataset=$1

num_transformer=$2 #-1 = All; 0 = cnn; 1 = gigaword; 2 = xsum; 3 = biomrc if first argument is transformers
                    # 0 = without summary; 1 = with summary for others datasets

mkdir -p led_inference/summaries
mkdir -p led_inference/evaluation

#If param1= transformers, we import the library from transformers. 
#Else, the dataset must be on the bucket transformers-lucia/tfg/datasets/ as a .csv

if [ $source_dataset = 'transformers' ]
then
    echo "Executing transformers model..."
    python3 led_inference/LED_inference.py transformers $num_transformer

    echo "Evaluating model..."
    python3 led_inference/scoring.py $num_transformer
    
else
    mkdir -p led_inference/datasets/

    echo "Executing $source_dataset dataset..."
    python3 led_inference/LED_inference.py $source_dataset $num_transformer
fi
