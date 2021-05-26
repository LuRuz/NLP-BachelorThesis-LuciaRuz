#!/bin/bash

# declare an array
arr=( "albert" "distilbert" "T5" )
# arr=( "pegasus" )

mkdir -p evaluation
mkdir -p summaries

for train_dataset in `seq 2 3`;
do
    # for loop that iterates over each element in arr
    for i in "${arr[@]}"
    do
        echo "Summarizing " $i "model on " $train_dataset
        python3 inference.py transformers $train_dataset $i
        
        echo "Evaluating " $i "model on " $train_dataset
        python3 scoring.py $train_dataset $i
    done
done

 

