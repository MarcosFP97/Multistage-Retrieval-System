#!/usr/bin/python3
import io
import json
from typing import List, Dict
from urllib import request, parse

import numpy as np
import pandas as pd
import typer
from torch import nn

from correctness.trainer_utils import ModelWrapper, ModelFunctionsWrapper
from evaluation.trec_data_loader import FOLDS_2021, TrecDataLoader

app = typer.Typer()

# Required for downloading qrels from gitlab
_TOKEN = "glpat-vgibykzCVvfe1YKX9Ehy"
_COMMIT_REF = "f4b2f22f"
get_url = lambda x: f"https://gitlab.citius.usc.es/api/v4/projects/1710/repository/files/" \
                    f"{parse.quote(x, safe='')}/raw?private_token={_TOKEN}&ref={_COMMIT_REF}"


# this class packs one RawEvaluator for helpful and harmful scores, and handles evaluation topics
class TrecRunScorer:
    def __init__(self, eval_topics: List[int]):
        helpful_dir = "evaluation/misinfo-resources-2021/qrels/2021-derived-qrels/misinfo-qrels-graded.helpful-only"
        harmful_dir = "evaluation/misinfo-resources-2021/qrels/2021-derived-qrels/misinfo-qrels-graded.harmful-only"
        self.helpful = RawCompatibilityEvaluator(get_url(helpful_dir), eval_topics)
        self.harmful = RawCompatibilityEvaluator(get_url(harmful_dir), eval_topics)
        self.eval_topics = eval_topics

    def __call__(self, run: Dict[str, List[str]], p: float = 0.95) -> (float, float, float):
        """
        Evaluate the run against the qrels.
        :param run: Dict 'topic' -> List of 'documentIDs'. Ordered list of documents for each topic
        :param p:
        :return: Tuple(helpful, harmful, helpful-harmful)
        """
        _, helpful_score = self.helpful.eval(run, p)
        _, harmful_score = self.harmful.eval(run, p)
        return helpful_score, harmful_score, helpful_score - harmful_score


# wrapper around original compatibility code, modified for fold topics evaluation
class RawCompatibilityEvaluator:
    def __init__(self,
                 qrel: str,
                 part_topics: List[int] = None,
                 normalize=True,
                 depth=1000):
        self._master_ideals, self.qrels = self._get_ideal_and_qrels(qrel, part_topics)
        self.normalize = normalize
        self.depth = depth

    @staticmethod
    def _get_ideal_and_qrels(qrels_path, part_topics: List[int]):
        ideal = {}
        qrels = {}
        try:
            file_object = open(qrels_path)
        except:
            file_object = io.StringIO(request.urlopen(qrels_path).read().decode('utf-8'))
        with file_object as qrelsf:
            for line in qrelsf:
                (topic, q0, docno, qrel) = line.rstrip().split()
                if part_topics is not None and int(topic) not in part_topics:
                    continue
                qrel = float(qrel)
                if qrel > 0.0:
                    if topic not in qrels:
                        ideal[topic] = []
                        qrels[topic] = {}
                    if docno in qrels[topic]:
                        if qrel > qrels[topic][docno]:  # if inutil en 2021
                            qrels[topic][docno] = qrel
                    else:
                        ideal[topic].append(docno)
                        qrels[topic][docno] = qrel
                    # assert ideal[topic] == list(qrels[topic].keys())
        return ideal, qrels

    def _rbo(self, run, ideal, p):
        run_set = set()
        ideal_set = set()
        score = 0.0
        normalizer = 0.0
        weight = 1.0
        for i in range(self.depth):
            if i < len(run):
                run_set.add(run[i])
            if i < len(ideal):
                ideal_set.add(ideal[i])
            score += weight * len(ideal_set.intersection(run_set)) / (i + 1)
            normalizer += weight
            weight *= p
        return score / normalizer

    @staticmethod
    def _idealize(run, ideal, qrels):
        """
        Given a run and a set of ideal documents, return the ideal ordering of the ideal documents,
        given the qrels and keeping the run ordering if qrels do not decide
        """
        rank = {}
        for i in range(len(run)):  # TODO: this step could be optimised
            rank[run[i]] = i
        ideal.sort(key=lambda docno: rank[docno] if docno in rank else len(run))
        ideal.sort(key=lambda docno: qrels[docno], reverse=True)
        return ideal

    def eval(self, run, p=0.95):
        run_ideals = {}
        count = 0
        total = 0.0
        scores = {}
        for topic in run:
            if topic in self._master_ideals:
                run_ideals[topic] = self._idealize(run[topic], self._master_ideals[topic].copy(), self.qrels[topic])
                score = self._rbo(run[topic], run_ideals[topic], p)
                if self.normalize:
                    best = self._rbo(run_ideals[topic], run_ideals[topic], p)
                    if best > 0.0:
                        score /= best
                    else:  # caso que no se da en 2021
                        score = best
                count += 1
                total += score
                # print(runid, topic, score, sep=',')
                # print('compatibility', topic, "{:.4f}".format(score), sep='\t')
                scores[topic] = score
        mean_score = total / count if count > 0 else 0.0
        return scores, mean_score


'''
This class implements a traditional model (Random Forest) to evaluate in batches all passages
'''


class SparseModel:
    def __init__(self, model: str, vectorizer: str):
        import pickle
        self.vectorizer = pickle.load(open(vectorizer, 'rb'))
        self.model = pickle.load(open(model, 'rb'))

    def __call__(self, inputs: List[str]) -> List[float]:
        predictions = []
        for passage in inputs:
            pred = list(self.model.predict_proba(self.vectorizer.transform([passage]))[0])[
                1]  ## we take the probability of being non-readable # classifier.predict_one(text)[1] #
            predictions.append(pred)

        return predictions


class ModelEvaluator:
    def __init__(self,
                 run_scorer: TrecRunScorer,
                 passages_df: pd.DataFrame,
                 extended_eval: bool = False):
        """
        Model evaluator, given a run scorer and a list of passages, evaluates the run against the qrels.
        :param run_scorer: The TrecRunScorer object
        :param extended_eval: If True, the evaluator will proccess each passage. If False, only the passages within the test fold topics.
                Useful for writing to csv output with all data, not useful for the compat score. Default False
        :param passages_df: pandas dataframe with the following columns: topic, docId, passage
        """
        self.extended_eval = extended_eval
        self.run_scorer = run_scorer
        assert {'topic', 'docId', 'passage'}.issubset(set(passages_df.columns.tolist()))
        self.b = passages_df
        if not self.extended_eval:
            self.b = self.b[
                self.b["topic"].isin(self.run_scorer.eval_topics)]  # only retain the topics in the test partition
        # self.b.loc[self.b.index[self.b.passage.notnull()], ['score_correct']] = -1.0  # set all scores to -1.0 when there is a passage
        self.passages = self.b[
            self.b.passage.notnull()].passage.tolist()  # first 100 rows have a passage and score_correct is not null, is -1.

    def __call__(self, scores: List[float],
                 column_name: str = 'score_correct',
                 write_run_csv: str = None
                 ) -> (float, float, float):
        run, run_pds = {}, []
        # self.eval_df['score_correct'] = scores  # aqui vienen las 1090
        self.b.loc[self.b.index[self.b.passage.notnull()], [column_name]] = scores
        for topic in self.b.topic.unique().tolist():
            df = self.b[self.b.topic == topic].sort_values(by=[column_name], ascending=False)
            run[str(topic)] = df.docId.tolist()
            if write_run_csv is not None:
                run_pds.append(df.copy())
        if write_run_csv is not None:
            pd.concat(run_pds).to_csv(write_run_csv, index=False, sep=" ")
        return self.run_scorer(run)


# 19_20_21 model_northern-glade-28/checkpoint-1000
# 19_20_partial21 model_desert-dew-27/checkpoint-1000
# 19_20 model_old-cantina-24/checkpoint-500
# 2020 manueldeprada/ctTREC-distillbert-correct-classifier-trec2020 dauntless glitter5 checkpoint 500
@app.command()
def eval_transformer_model(
        model: str = 'readability/bert-readability/1655735485760',
        fold: int = 1,
        partition: str = 'test',
        write_run_csv: str = 'runs/readability',
):
    assert partition in ['train', 'valid', 'test']
    print(f'Loading model {model}... ')
    model = ModelWrapper(model, model_fcts=ModelFunctionsWrapper(nn.BCELoss(),  # nn.CrossEntropyLoss(weight=None),
                                                                 num_labels=1,
                                                                 correct_label=0))
    print(f'Loading scorer and ModelEvaluator for {partition}... ')
    scorer = TrecRunScorer(FOLDS_2021[fold][partition])
    # data_df = TrecDataLoader.get_2021_df()  # esta liña recupera todos os pasaxes nos qrels de correctitude, de todos os folds
    data_df = TrecDataLoader.get_bm25_run_df(run_path='evaluation/2021_bm25_base_run.txt')
    evaluator = ModelEvaluator(run_scorer=scorer, passages_df=data_df)
    if write_run_csv is not None:
        print(f"Evaluating to {write_run_csv}_{partition.upper()}_{fold}.csv")
        write_run_csv = f"{write_run_csv}_{partition.upper()}_{fold}.csv"
    passages_to_score = evaluator.passages
    scores_for_passages = model(passages_to_score)
    compat_score = evaluator(scores=scores_for_passages, column_name='score_readability', write_run_csv=write_run_csv)
    print(compat_score)


@app.command()
def eval_rf_model(
        model: str = 'credibility/results/cred_model.pickle',
        vectorizer: str = 'credibility/results/vectorizer.pickle',
        fold: int = 4,
        partition: str = 'test',
        write_run_csv: str = 'runs/citius.credibility',
):
    assert partition in ['train', 'valid', 'test']
    print(f'Loading model {model}... ')
    model = SparseModel(model, vectorizer)
    print(f'Loading scorer and ModelEvaluator for {partition}... ')
    scorer = TrecRunScorer(FOLDS_2021[fold][partition])
    # data_df = TrecDataLoader.get_2021_df()  # esta liña recupera todos os pasaxes nos qrels de correctitude, de todos os folds
    data_df = TrecDataLoader.get_bm25_run_df(run_path='evaluation/monot5descr-2022.txt')
    evaluator = ModelEvaluator(run_scorer=scorer, passages_df=data_df)
    if write_run_csv is not None:
        print(f"Evaluating to {write_run_csv}_{partition.upper()}_{fold}.csv")
        write_run_csv = f"{write_run_csv}_{partition.upper()}_{fold}.csv"
    passages_to_score = evaluator.passages
    print("!!!! Passage:", passages_to_score[0])
    scores_for_passages = model(passages_to_score)
    compat_score = evaluator(scores=scores_for_passages, column_name='score_credibility', write_run_csv=write_run_csv)
    print(compat_score)


# metodo para testear os scores topico a topico
@app.command()
def test_topics(
        model: str = 'correctness/model_northern-glade-28/checkpoint-1000',
        write_run_csv: str = None  # 'runs/4_signals/correct/PERTOPIC_distillbert-correct-classifier-2019_2020_2021',
):
    print(f'Loading model {model}... ', end='', flush=True)
    # model = ModelWrapper(model)
    print('done.')
    df_2021 = TrecDataLoader.get_2021_df()
    # valid_df_2021 = df_2021[df_2021['topic'].isin(Trec2021Partitions.VALID.value)]

    for fold in [1, 2, 3]:
        for part in ['train', 'valid', 'test']:
            partition_list = FOLDS_2021[fold][part]
            df = df_2021[df_2021['topic'].isin(FOLDS_2021[fold][part])]["labels"].tolist()
            print(f'Topics from f{fold}p{part}. 0s:{df.count(0)}, 1s: {df.count(1)}, ratio1s: {df.count(1) / len(df)}')

    print("now by topic unique values")
    for topic in df_2021['topic'].unique():
        df = df_2021[df_2021['topic'].isin([topic])]["labels"].tolist()
        print(f'Topics from t{topic}. 0s:{df.count(0)}, 1s: {df.count(1)}, ratio1s: {df.count(1) / len(df)}')
    total_df = df_2021["labels"].tolist()
    print(f'Total: 0s:{total_df.count(0)}, 1s: {total_df.count(1)}, ratio1s: {total_df.count(1) / len(total_df)}')
    return
    for partition in ['train', 'valid', 'test']:
        partition_list = FOLDS_2021[1][part]
        print(f'Topics from {partition.name}:')
        for topic in partition_list:
            scorer = TrecRunScorer([topic])
            evaluator = ModelEvaluator(run_scorer=scorer, passages_df=df_2021)
            if write_run_csv is not None:
                print(f"Evaluating to {write_run_csv}_topic{topic}.csv")
                write_run_csv = f"{write_run_csv}_topic{topic}.csv"
            print(
                f"topic {topic}: {evaluator(scores=model(evaluator.passages, quiet=True), write_run_csv=write_run_csv)}")


'''
This method evaluates TREC format runs that don't use a model for evaluation
It replaces the previous csv method 
'''


@app.command()
def evaluate_run(
        run: str = 'runs/T5_gpt_and_se.txt',
        fold: int = 4,
        partition: str = 'test',
        field: str = 'score'
):
    assert partition in ['train', 'valid', 'test']
    df = pd.read_csv(run, sep=' ', names=["topic", "Q0", "docId", "rank", "score", "tag"])

    ######################### SCORE FUSION (TO-DO) #############################################
    df = df[["topic", "docId", field]]
    grouped = df.groupby(df.topic)
    run = {}
    for topic, g in grouped:
        topic = str(topic)
        df1 = g[:100]
        df2 = g[100:]
        df1 = df1.sort_values(by=[field], ascending=False)
        topic_df = pd.concat([df1, df2])
        topic_df[field] = len(topic_df) - np.arange(len(topic_df))  # inverse ranking position
        if topic not in run:
            run[topic] = topic_df['docId'].tolist()
    ################################################################################################

    scorer = TrecRunScorer(FOLDS_2021[fold][partition])
    helpful, harmful, total = scorer(run)
    # print(*scores.items(), sep='\n')
    print('helpful\tharmful\ttotal')
    print(f'{helpful:.4f}\t{harmful:.4f}\t{total:.4f}')


'''
This method implements the fusion of two or more signals with CombSUM strategy, in a weighted and unweighted manner
'''


@app.command()
def score_fusion(
        input_csv: str = 'runs/citius.readability.txt',
        second_csv: str = None,
        signals: List[str] = ['score_passage', 'score_credibility', 'score_readability'],
        weights: List[float] = [0.95, 0.025, 0.025],
        output_csv: str = 'runs/citius.readability.merged',
):
    input_df = pd.read_csv(input_csv, sep=' ')
    df = input_df[["topic", "docId", signals[0], signals[1], signals[2]]]

    # if second_csv: ### 3-signal fusion
    #     second_df = pd.read_csv(second_csv, sep=' ')
    #     second_df = second_df[["topic", "docId", signals[2]]]
    #     df = pd.merge(df, second_df, how="left", on=["topic", "docId"])

    grouped = df.groupby(df.topic)
    run = {}
    list_of_dfs = []
    for topic, g in grouped:
        topic = str(topic)
        df1 = g[:100]
        df2 = g[100:]
        df1[signals[0]] = np.exp(df1[signals[0]])

        # if second_csv: #### 3-signal fusion
        df1['score_avg'] = (weights[0] * df1[signals[0]] + weights[1] * df1[signals[1]] + weights[1] * df1[
            signals[2]]) / 3
        # else:
        #     df1['score_avg'] = (weights[0]*df1[signals[0]] + weights[1]*df1[signals[1]]) / 2

        df1 = df1.sort_values(by=['score_avg'], ascending=False)
        topic_df = pd.concat([df1, df2])
        topic_df['score_avg'] = len(topic_df) - np.arange(len(topic_df))  # inverse ranking position
        list_of_dfs.append(topic_df)
        if topic not in run:
            run[topic] = topic_df['docId'].tolist()
    pd.concat(list_of_dfs).to_csv(f"{output_csv}.txt")
    # input_csv = input_csv.split('_')
    # partition = input_csv[1].lower()
    # fold = int(input_csv[2].replace('.csv',''))
    # scorer = TrecRunScorer(FOLDS_2021[fold][partition])
    # helpful, harmful, total = scorer(run)
    # # print(*scores.items(), sep='\n')
    # print('helpful\tharmful\ttotal')
    # print(f'{helpful:.4f}\t{harmful:.4f}\t{total:.4f}')


@app.command()
def score_fusion_manuel(
        input_csv: str = 'runs/citius.readability.txt',
        second_csv: str = None,
        signals: List[str] = ['score_monoT5', 'score_correct'],
        weights: List[float] = [0.95, 0.025, 0.025],
        output_csv: str = 'runs/bert.monot5.merged',
        fold: int = 1,
        partition: str = 'valid',
):
    input_df = pd.read_csv(input_csv, sep=' ')
    columns_list = ["topic", "docId"] + signals
    df = input_df[columns_list]

    # if second_csv: ### 3-signal fusion
    #      second_df = pd.read_csv(second_csv, sep=' ')
    #      second_df = second_df[["topic", "docId", signals[2]]]
    #      df = pd.merge(df, second_df, how="left", on=["topic", "docId"])

    grouped = df.groupby(df.topic)
    run = {}
    list_of_dfs = []
    for topic, g in grouped:
        topic = str(topic)
        df1 = g[:100]
        df2 = g[100:]
        df1[signals[0]] = np.exp(df1[signals[0]])

        # if second_csv: #### 3-signal fusion
        # df1['score_avg'] = (weights[0] * df1[signals[0]] + weights[1] * df1[signals[1]] + weights[1] * df1[signals[2]]) / 3
        # else:
        df1['score_avg'] = (weights[0]*df1[signals[0]] + weights[1]*df1[signals[1]]) / 2

        df1 = df1.sort_values(by=['score_avg'], ascending=False)
        topic_df = pd.concat([df1, df2])
        topic_df['score_avg'] = len(topic_df) - np.arange(len(topic_df))  # inverse ranking position
        list_of_dfs.append(topic_df)
        if topic not in run:
            run[topic] = topic_df['docId'].tolist()
    pd.concat(list_of_dfs).to_csv(f"{output_csv}.txt")
    # input_csv = input_csv.split('_')
    # partition = input_csv[1].lower()
    # fold = int(input_csv[2].replace('.csv',''))
    scorer = TrecRunScorer(FOLDS_2021[fold][partition])
    helpful, harmful, total = scorer(run)
    # # print(*scores.items(), sep='\n')
    print('helpful\tharmful\ttotal')
    print(f'{helpful:.4f}\t{harmful:.4f}\t{total:.4f}')


if __name__ == '__main__':
    app()
