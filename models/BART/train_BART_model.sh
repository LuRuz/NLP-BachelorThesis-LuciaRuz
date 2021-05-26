#!/bin/bash

num_transformer=$1 #-1 = All; 0 = cnn; 1 = gigaword; 2 = xsum; 3 = biomrc

#Install libraries
pip install -r requirements.txt


mkdir -p ./bart_model/models

if [ $num_transformer -eq -1 ]
then
    echo "EXECUTING ALL"

    for train_dataset in `seq 0 3`;
    do
        echo "Executing dataset number $train_dataset"
        python3 ./bart_model/bart_training.py $train_dataset
    done

elif [ "$num_transformer" -gt -1 ] && [ "$num_transformer" -lt 4 ]
then
        echo "Executing dataset number $num_transformer"
        python3 ./bart_model/bart_training.py $num_transformer
else
    echo "Model not valid, try again."
fi

