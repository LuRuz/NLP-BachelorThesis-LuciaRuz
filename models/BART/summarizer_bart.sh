#!/bin/bash

pip install -r requirements.txt

# To call this funcition ./summarizer.sh param1 where param1 is the name of the dataset to summarize
source_dataset=$1

num_transformer=$2 #-1 = All; 0 = cnn; 1 = gigaword; 2 = xsum; 3 = biomrc if first argument is transformers
                    # model to make inferente for others datasets
                    
with_or_without_sum=$3 #for other datasets, 0 means no source summary; 1 with summary

mkdir -p bart_model/models
mkdir -p bart_model/summaries
mkdir -p bart_model/evaluation

if [ $source_dataset = 'transformers' ]
then
    #List with all posibilities to evaluate
    transformer_datasets=(cnn_dailymail gigaword xsum biomrc)
    
    if [ $num_transformer -eq -1 ]
    then
        echo "EXECUTING ALL"

        for train_dataset in `seq 0 3`;
        do
            echo "Executing transformers model..."
            python3 ./bart_model/bart_inference.py transformers $train_dataset -1

            echo "Evaluating model..."
            python3 ./bart_inference/bart_scoring.py ${transformer_datasets[$train_dataset]} 
        done

    #Execute only one dataset
    elif [ "$num_transformer" -gt -1 ] && [ "$num_transformer" -lt 4 ]
    then     
        echo "Executing transformers model..."
        python3 ./bart_model/bart_inference.py transformers $num_transformer -1

#         echo "Evaluating model..."
#         python3 ./bart_inference/bart_scoring.py ${transformer_datasets[$num_transformer]} 
    
    else
        echo "Incorrect dataset"
    fi

# Inference with a diferent dataset
else
    mkdir -p ./bart_model/datasets/

    echo "Executing $source_dataset dataset..."
    python3 ./bart_model/bart_inference.py $source_dataset $num_transformer $with_or_without_sum
    
    if [ $with_or_without_sum -eq 1 ]
    then
        echo "Evaluating model..."
        python3 ./bart_inference/bart_scoring.py $source_dataset
    else
        echo "We can't evaluate our inference..."
    fi
fi
