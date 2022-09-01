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
os.environ["JAVA_HOME"] = "/opt/citius/modules/software/Java/11.0.2"
import xml.etree.ElementTree as ET
from pyserini.search.lucene import LuceneSearcher

def search(topic, field, searcher, out_dir):
    text = topic.find(field).text
    number = topic.find("number").text
    # stance = topic.find("stance").text
    print(text)
    hits = searcher.search(text, 1000)

    results = []
    count = 1
    #  the first thousand hits:
    for i in range(0, len(hits)):
        json_doc = json.loads(hits[i].raw)
        dd = {"topic": number, "Q0": "Q0", "docId": hits[i].docid, "rank": count, "score": hits[i].score, "text": json_doc['text'], "description": text, "tag": "BM25"}
        results.append(dd)
        count +=1
    
    df = pd.DataFrame(results)
    df.set_index('topic', inplace=True)
    df.to_csv(f'{out_dir}/topic'+number+'.csv')


def main(field: str = 'question',
         index: str = 'indexes/C4/',
         topics_file: str = 'trec-pipeline/evaluation/misinfo-2022-topics.xml',
         out_dir: str = 'trec-pipeline/runs/trec-2022/bm25'):
    searcher = LuceneSearcher(index)
    print("Index loaded")
    print("=============")
    root = ET.parse(topics_file).getroot()
    for topic in root.findall('topic'):
        search(topic, field, searcher, out_dir)


if __name__ == "__main__":
    typer.run(main)
