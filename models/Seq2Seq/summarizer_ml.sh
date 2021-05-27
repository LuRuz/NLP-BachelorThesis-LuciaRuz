#!/bin/bash

pip install -r requirements.txt

# To call this funcition ./summarizer.sh param1 where param1 is the name of the dataset to summarize
source_dataset=$1

num_transformer=$2 #-1 = All; 0 = cnn; 1 = gigaword; 2 = xsum; 3 = biomrc

mkdir -p ml_aprox/summaries
mkdir -p ml_aprox/evaluation

#If param1= transformers, we import the library from transformers. Else, the dataset must be on the bucket transformers-lucia/tfg/datasets/ as a folder with multiples files .txt
echo "Transformers dataset has been selected"

if [ $source_dataset = 'transformers' ]
then
    echo "Executing transformers model"
    python3 ml_aprox/ml_inference.py transformers $num_transformer
    
    echo "Evaluating model"
    python3 ml_aprox/ml_scoring.py $num_transformer
    
else
    echo "NO IMPLEMENTADO"
    echo "Executing $source_dataset"
    python3 ml_aprox/ml_inference.py source_dataset 10
fi

rm -rf ml-aprox/summaries