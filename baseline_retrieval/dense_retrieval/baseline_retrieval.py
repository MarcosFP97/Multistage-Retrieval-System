from operator import indexOf
import os
import re
import json
import argparse
import threading
import pandas as pd
import time
import nltk
import json
import typer

nltk.download('stopwords')

from nltk.corpus import stopwords
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
import xml.etree.ElementTree as ET
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder

def search(topic, field, searcher, out_dir):
    text = topic.find(field).text
    number = topic.find("number").text
    stance = topic.find("stance").text
    print(text)
    hits = searcher.search(text, 1000)

    results = []
    count = 1
    #  the first thousand hits:
    for i in range(0, len(hits)):
        json_doc = json.loads(hits[i].raw)
        dd = {"topic": number, "Q0": "Q0", "docId": hits[i].docid, "rank": count, "score": hits[i].score, "text": json_doc['text'], "description": text, "stance": stance, "tag": "DenseRetrieval"}
        results.append(dd)
        count +=1
    
    df = pd.DataFrame(results)
    df.set_index('topic', inplace=True)
    df.to_csv(f'{out_dir}/topic'+number+'.csv')


def main(field: str = 'description',
         index: str = 'indexes/C4/',
         topics_file: str = 'experiments/trec-pipeline/2021-topics.xml',
         out_dir: str = 'experiments/trec-pipeline/runs/test_run/dense_retrieval'):
    encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
    searcher = FaissSearcher.from_prebuilt_index(index, encoder)
    print("Index loaded")
    print("=============")
    root = ET.parse(topics_file).getroot()
    for topic in root.findall('topic'):
        search(topic, field, searcher, out_dir)


if __name__ == "__main__":
    typer.run(main)