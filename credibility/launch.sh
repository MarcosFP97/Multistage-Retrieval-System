#!/bin/bash 
docker build -t cred_alg .
docker run -d --name cred_alg_1 -e PYTHONUNBUFFERED=0 -v /home/marcos.fernandez.pichel/trec-pipeline-2022/credibility/clef2018collection:/usr/src/app/pygaggle/clef2018collection -v /home/marcos.fernandez.pichel/trec-pipeline-2022/credibility/results:/usr/src/app/pygaggle/results --gpus all cred_alg

