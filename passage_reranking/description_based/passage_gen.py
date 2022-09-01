from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Union, Optional

import nltk
import pandas as pd
import typer
from pandas import DataFrame
from tqdm import tqdm

nltk.download('punkt', quiet=True)

app = typer.Typer()


def window(iterable, n=6, m=3):
    """
    This function creates a sliding window over text
    :param iterable:
    :param n:
    :param m:
    """
    if m == 0:  # otherwise infinte loop
        raise ValueError("Parameter 'm' can't be 0")
    lst = list(iterable)
    i = 0

    if i + n < len(lst):
        while i + n < len(lst):
            yield ' '.join(lst[i:i + n])
            i += m

    else:
        yield ' '.join(lst)


def get_passages_for_dfs(dfs: Union[List[DataFrame], DataFrame],
                         model_name: str = 'castorini/monot5-base-med-msmarco',
                         tokenizer_name: str = 't5-base',
                         extra_cols: Optional[List[str]] = None) -> DataFrame:
    if isinstance(dfs, list):
        df = pd.concat(dfs)
    else:
        df = dfs
    if extra_cols is None:
        extra_cols = []
    from langdetect import detect
    from pygaggle.rerank.base import Query, Text
    from pygaggle.rerank.transformer import MonoT5

    # model = MonoT5.get_model(model_name)
    tokenizer = MonoT5.get_tokenizer(tokenizer_name)
    # reranker = MonoT5(model, tokenizer)
    reranker = MonoT5(pretrained_model_name_or_path=model_name, tokenizer=tokenizer)
    # queries = pd.read_csv('./pygaggle/predicted_sentences_top1.txt')
    df["score_monot5"] = 0.0
    df['passage'] = ""
    print(f"Generating passages for {len(df)} documents...")
    for i, (query, text) in tqdm(enumerate(zip(df['description'].values, df['text'].values)), total=len(df)):
        lines = (line.strip() for line in text.splitlines())  # break into lines and remove leading and trailing space on each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # break multi-headlines into a line each
        doc = ' '.join(chunk for chunk in chunks if chunk)  # drop blank lines
        doc = doc.replace('\\n', ' ').replace('\\r', '').replace('\\t', '').replace('\\', '')  # remove end-line characters

        try:
            lang = detect(doc)
        except:
            lang = "error"

        if lang == "en":
            doc = nltk.sent_tokenize(doc)
            doc_passages = window(doc, 6, 3)  # len = 6, stride = 3
            texts = [Text(p.encode("utf-8", errors="replace").decode("utf-8"), None, 0) for p in doc_passages]

            query = Query(query)
            reranked = reranker.rerank(query, texts)
            reranked.sort(key=lambda x: x.score, reverse=True)
            if reranked:
                passage = reranked[0].text  # we select the top 1 reranked passage and score (= most relevant passage for the query)
                score = reranked[0].score  # mean([reranked[pos_text].score for pos_text in range(len(reranked))]) #
                df.iloc[i, df.columns.get_loc('score_monot5')] = score
                df.iloc[i, df.columns.get_loc('passage')] = passage
    cols = ['topic', 'docId', 'score_monot5', 'passage'] + extra_cols
    return df[cols]


@app.command()
def generate_passages_from_bm25(model_name: str = 'castorini/monot5-base-med-msmarco',
                                tokenizer_name: str = 't5-base',
                                bm25_runs_dir: str = 'trec-pipeline/runs/trec-2022',
                                out_file: str = 'runs/monot5/monot5descr-2022.csv'):
    directory = Path(bm25_runs_dir)
    dfs = [pd.read_csv(file) for file in directory.glob('*.csv')]
    df = get_passages_for_dfs(dfs, model_name, tokenizer_name, extra_cols=['score'])
    df.to_csv(Path(out_file), index=False)


def get_topics(topics_file: str = 'evaluation/misinfo-resources-2021/topics/misinfo-2021-topics.xml'):
    root = ET.parse(topics_file).getroot()
    topics = {}
    for topic in root.findall('topic'):
        description = topic.find("description").text
        number = topic.find("number").text
        stance = topic.find("stance").text
        topics[int(number)] = description
    return topics


@app.command()
def generate_passages_from_qrels(model_name: str = 'castorini/monot5-base-med-msmarco',
                                 collection_index: str = '../../indexes/C4/',
                                 tokenizer_name: str = 't5-base',
                                 qrels_file: str = 'evaluation/misinfo-resources-2021/qrels/qrels-35topics.txt',
                                 out_file: str = 'runs/2_passages_for_all_qrels_docs_2021.csv'):
    qrels_df = pd.read_csv(qrels_file, sep=' ', names=['topic', 'Q0', 'docId', 'useful', 'supportive', 'credible'])
    topics = get_topics()
    qrels_df['description'] = qrels_df.apply(lambda row: topics[int(row['topic'])], axis=1)
    from pyserini.search import SimpleSearcher
    searcher = SimpleSearcher(collection_index)
    print(f"Retrieving {len(qrels_df)} documents...")
    tqdm.pandas()
    qrels_df['text'] = qrels_df.progress_apply(lambda row: searcher.doc(row['docId']).raw(), axis=1)
    df = get_passages_for_dfs(qrels_df, model_name, tokenizer_name)
    df.to_csv(Path(out_file), index=False)


def join_qrels_passages(qrels_file, passages_file):
    df_passages = pd.read_csv(Path(passages_file))
    df_passages = df_passages[['topic', 'docId', 'passage']]
    df_qrels = pd.read_csv(Path(qrels_file), sep=' ', names=['topic', 'Q0', 'docId', 'relevance'])
    df_qrels = df_qrels[['topic', 'docId']]
    df_qrels_with_passage = df_qrels.merge(df_passages, how='inner', on=['topic', 'docId'])
    df_qrels_without_passage = df_qrels.merge(df_passages, how='outer', indicator=True)
    df_qrels_without_passage = df_qrels_without_passage[df_qrels_without_passage['_merge'] == 'left_only'].drop('_merge', axis=1)
    print(f"{len(df_qrels_with_passage)/len(df_qrels)*100}% of the qrels have passages in {qrels_file}")
    return df_qrels_with_passage


@app.command()
def map_passages_to_qrels(passages_file: str = 'runs/2_passages_for_all_qrels_docs_2021.csv',
                          correct_qrels_file: str = 'evaluation/misinfo-resources-2021/qrels/2021-derived-qrels/misinfo-qrels-binary.useful-correct',
                          incorrect_qrels_file: str = 'evaluation/misinfo-resources-2021/qrels/2021-derived-qrels/misinfo-qrels-binary.incorrect',
                          output_file: str = 'correctness/training_data_2021_correct.csv'):
    correct_with_passage = join_qrels_passages(correct_qrels_file, passages_file)
    correct_with_passage['correctness'] = 1
    incorrect_with_passage = join_qrels_passages(incorrect_qrels_file, passages_file)
    incorrect_with_passage['correctness'] = 0
    df = pd.concat([correct_with_passage, incorrect_with_passage])
    df.to_csv(Path(output_file), index=False, header=True, sep=' ')


if __name__ == '__main__':
    app()
