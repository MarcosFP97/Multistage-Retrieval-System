#!/bin/bash

<<comment
   - This script must be run inside Anserini (https://github.com/castorini/anserini) compiled folder
   - Args: 
     $1: specific Collection Class (responsible for reading the text from disk in a particular format)
     $2: path to input dataset
     $3: path where index must be saved
     $4: specific Generator Class (responsible for generating a Lucene document, e.g., populating the right fields to index)
     $5: experiment name (= log file name)
comment

nohup sh target/appassembler/bin/IndexCollection -collection $1 -input $2 -index $3 -generator $4 -threads 44 -storePositions -storeDocvectors -storeRaw >& logs/$5 &