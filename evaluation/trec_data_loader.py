from enum import Enum
from pathlib import Path
from typing import Union, List

import datasets
import pandas as pd
from datasets import Value, Features, ClassLabel, Dataset, DatasetInfo, DatasetDict

FOLDS_2021 = {
    1: {
        "train": [144, 112, 140, 143, 109, 137, 131, 102, 132, 136, 118, 146, 133, 117, 101],
        "valid": [145, 139, 149, 115, 134, 107, 105, 103, 122, 127],
        "test": [111, 108, 104, 128, 110, 121, 106, 129, 120, 114]
    },
    2: {
        "train": [129, 127, 131, 110, 132, 149, 118, 117, 133, 122, 109, 108, 144, 139, 112],
        "valid": [128, 143, 111, 106, 102, 114, 103, 120, 104, 137],
        "test": [101, 136, 105, 121, 115, 107, 146, 134, 140, 145]
    },
    3: {
        "train": [110, 128, 132, 107, 121, 111, 115, 117, 112, 139, 149, 145, 102, 104, 134],
        "valid": [103, 120, 137, 140, 114, 146, 143, 105, 136, 131],
        "test": [122, 118, 106, 109, 101, 133, 108, 127, 129, 144]
    },
    4: {
        "test": [110, 128, 132, 107, 121, 111, 115, 117, 112, 139, 149, 145, 102, 104, 134, 103, 120, 137, 140, 114, 146, 143, 105, 136, 131, 122, 118, 106, 109, 101, 133, 108, 127, 129, 144]
    }
}


# Unified dataset loader for TREC experiments
class TestSet(Enum):
    TREC_2021_FOLDS = -3
    PARTITION = 0
    TREC_2019 = 1
    TREC_2020 = 2
    TREC_2021 = 3


class TrainSet(Enum):
    TREC_2019 = 1
    TREC_2020 = 2
    TREC_2021 = 3


class TrecDataLoader:

    def get_hf_dataset(self, *, train_set: Union[TrainSet, List[TrainSet]],
                       test_set: TestSet,
                       test_size: float = 0.05,
                       fold: int = 1) -> DatasetDict:
        """ Loads the dataset for the given train and test sets. Train sets avaliable are:  TREC_2019, TREC_2020, TREC_2021.
        Test sets avaliable are: TOPICS_2021 (selected topics), PARTITION (take a split of the train set), TREC_2019, TREC_2020, TREC_2021.
        If test_set is TOPICS_2021, test_size is ignored. If test_set is also used as train_set, a separate split will be taken.

        :param train_set: the train set to load. If a list is given, the union of all train sets is returned.
        :param test_set:  the test set to load
        :param test_size: the percentage of the test set
        :return: a dictionary with the train and test datasets
        """

        def dt_from_df(name, df, labels=True):
            features = Features({'topic': Value('int64'),
                                 'docId': Value('string'),
                                 'topic_description': Value('string'),
                                 'passage': Value('string')})
            if labels:
                features['labels'] = ClassLabel(names=['incorrect', 'correct'])

            df.reset_index(drop=True, inplace=True)
            return Dataset.from_pandas(df, info=DatasetInfo(features=features, description=name))

        if isinstance(train_set, TrainSet):
            train_set = [train_set]

        train_dfs = [(t.name, self._get_df(t)) for t in train_set if t.value != abs(test_set.value)]  # if test includes 2021, abs is needed
        train_dts = [dt_from_df(name, df) for name, df in train_dfs]

        if test_set.value in [t.value for t in train_set]:  # case test_set = split of dataset in train_set
            test_df = self._get_df(TrainSet(test_set.value))
            test_dt = dt_from_df(test_set.name, test_df)
            final_dt_dict = test_dt.train_test_split(test_size=test_size, seed=test_set.value)
            final_dt_dict["train"] = datasets.concatenate_datasets(train_dts + [final_dt_dict["train"]]).shuffle(seed=test_set.value)
        elif test_set == TestSet.TREC_2021_FOLDS:  # case test_set = selected topics of 2021
            df_2021 = self._get_df(TrainSet.TREC_2021)
            train_df_2021 = df_2021[df_2021['topic'].isin(FOLDS_2021[fold]['train'])]
            valid_df_2021 = df_2021[df_2021['topic'].isin(FOLDS_2021[fold]['valid'])]
            test_df_2021 = df_2021[df_2021['topic'].isin(FOLDS_2021[fold]['test'])]
            if TrainSet.TREC_2021 in train_set:
                train_dts.append(dt_from_df(f"TREC_2021_TRAIN_FOLD{fold}", train_df_2021))
            valid_dt = dt_from_df(f"TREC_2021_VALID_FOLD{fold}", valid_df_2021)
            test_dt = dt_from_df(f"TREC_2021_TEST_FOLD{fold}", test_df_2021)
            final_dt_dict = DatasetDict({"train": datasets.concatenate_datasets(train_dts).shuffle(seed=1), "valid": valid_dt, "test": test_dt})
        elif test_set == TestSet.PARTITION:  # case test_set = split of whole selection of datasets
            final_dt_dict = datasets.concatenate_datasets(train_dts).train_test_split(test_size=test_size, seed=1)
        else:  # case where train and test are different datasets. i.e train=2020, test=2019
            test_dt = dt_from_df(test_set.name, self._get_df(TrainSet(test_set.value))).train_test_split(test_size=test_size,
                                                                                                         seed=test_set.value)["test"]
            train_dt = datasets.concatenate_datasets(train_dts).shuffle(seed=test_set.value)
            final_dt_dict = DatasetDict({"train": train_dt, "test": test_dt})
        return final_dt_dict

    def _get_df(self, t: TrainSet):
        switch = {
            TrainSet.TREC_2019: self.get_2019_df,
            TrainSet.TREC_2020: self.get_2020_df,
            TrainSet.TREC_2021: self.get_2021_df,
        }
        return switch.get(t, lambda: "Invalid train set")()

    @staticmethod
    def get_bm25_run_df(run_path: Union[str, Path] = 'evaluation/2021_bm25_base_run.txt'):
        from numpy import nan
        run_path = Path(run_path)
        b = pd.read_csv(run_path, sep=" ")
        b = b[["topic", "docId", "passage", "score", "score_passage"]]
        b.rename(columns={'score': 'score_bm25', 'score_passage': 'score_monoT5'}, inplace=True)
        for topic in b.topic.unique().tolist():
            topic_i = b.index[b.topic == topic]
            g = b.loc[topic_i]  # group of passages of the topic. this is an index view in the df, not a copy
            b.loc[topic_i.intersection(g.index[100:]), ['score_monoT5']] = nan  # remove misplaced bm25 score from the monot5 column
            # b.loc[topic_i.intersection(g.index[:100]), ['score_correct']] = -1.0  # set the correctness score to -1 for the first 100 passages
        return b

    @staticmethod
    def get_2019_df(csv_path: str = "correctness/training_data_2019_correct.csv"):
        trec_2019_passages = Path(csv_path)
        passages = pd.read_csv(trec_2019_passages, sep=' ')  # names=['topic', 'Q0', 'docId', "correctness", 'passage', 'score_passages', 'score']
        passages = passages[["topic", "docId", "passage", "correctness"]]  # 4148
        repes_malos = passages.drop_duplicates(subset=["topic", "passage", "correctness"], keep=False)  # quito repes en las 3
        repes_malos = repes_malos[repes_malos.duplicated(subset=["topic", "passage"], keep=False)]  # no repes en 3 cols pero si topic & passage! (0)
        passages = pd.merge(passages, repes_malos, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        passages.drop_duplicates(subset=["topic", "passage"], keep="first", inplace=True)  # quedan 3951
        passages.rename(columns={'correctness': 'labels'}, inplace=True)
        passages.dropna(inplace=True)  # cae 1
        descriptions = TrecDataLoader.get_topics('evaluation/2019topics.xml')
        passages['topic_description'] = passages.topic.apply(lambda x: descriptions[int(x)])
        return passages

    @staticmethod
    def get_2021_df(csv_path: str = "correctness/training_data_2021_correct.csv"):
        trec_2021_passages = Path(csv_path)
        passages = pd.read_csv(trec_2021_passages, sep=' ')  # names=['topic', 'Q0', 'docId', "correctness", 'passage', 'score_passages', 'score']
        passages = passages[["topic", "docId", "passage", "correctness"]]  # 4556
        repes_malos = passages.drop_duplicates(subset=["topic", "passage", "correctness"], keep=False)  # quito repes en las 3
        repes_malos = repes_malos[repes_malos.duplicated(subset=["topic", "passage"], keep=False)]  # no repes en 3 cols pero si topic & passage! (8)
        passages = pd.merge(passages, repes_malos, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)  # 4548
        passages.drop_duplicates(subset=["topic", "passage"], keep="first", inplace=True)  # quedan 3866
        passages.rename(columns={'correctness': 'labels'}, inplace=True)
        passages.dropna(inplace=True)  # no cae ninguno
        descriptions = TrecDataLoader.get_topics()
        passages['topic_description'] = passages.topic.apply(lambda x: descriptions[int(x)])
        return passages

    @staticmethod
    def get_2020_df(csv_path: str = "correctness/training_data_2020_correct.csv"):
        # trec_home = Path("../")
        # trec_2020_passages = trec_home / "correctness/training_data_2020_correct.csv"
        trec_2020_passages = Path(csv_path)

        # trec_2020_passages = trec_home / "runs/2_monoT5_descr_adhoc_100_2020.txt"
        # trec_2020_correct = trec_home / "evaluation/misinfo-resources-2020/qrels/2020-derived-qrels/misinfo-qrels-binary.useful-correct"
        # trec_2020_incorrect = trec_home / "evaluation/misinfo-resources-2020/qrels/2020-derived-qrels/misinfo-qrels-binary.incorrect"

        # passages = pd.read_csv(trec_2020_passages, sep=' ', names=['topic', 'Q0', 'docId', "rank", 'score', 'score_descr', 'passage', 'DescrAdhoc'])
        # passages = passages[["docId", "passage"]]
        # passages = passages.dropna(subset=["passage"])
        passages = pd.read_csv(trec_2020_passages, sep=' ')  # names=['topic', 'Q0', 'docId', "correctness", 'passage', 'score_passages', 'score']
        passages = passages[["topic", "docId", "passage", "correctness"]]  # 4849

        # correct = pd.read_csv(trec_2020_correct, sep=' ', names=['topic', 'ignore', 'docId', 'correct'])
        # correct = correct[["docId", "topic"]]
        # correct.drop_duplicates(keep='first', inplace=True)

        # passages_correct = passages[passages["correctness"] == 1]
        # get passages whose columns docId and topic are in the correct set
        # passages_correct = passages_correct_1[["docId", "topic"]].merge(correct, on=["docId", "topic"], how="inner")

        # passages.drop_duplicates(keep='first', inplace=True)  # 4849 -> 4727

        # duplicated_passages = passages[passages.duplicated(subset=["topic", "passage"], keep=False)]
        repes_malos = passages.drop_duplicates(subset=["topic", "passage", "correctness"], keep=False)  # quito repes en las 3
        repes_malos = repes_malos[repes_malos.duplicated(subset=["topic", "passage"], keep=False)]  # no repes en 3 cols pero si topic & passage! (8)
        passages = pd.merge(passages, repes_malos, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)  # 4841
        passages.drop_duplicates(subset=["topic", "passage"], keep="first", inplace=True)  # quedan 4712
        # rename correctness column to labels
        passages.rename(columns={'correctness': 'labels'}, inplace=True)

        # incorrect = pd.read_csv(trec_2020_incorrect, sep=' ', names=['topic', 'ignore', 'docId', 'correct'])
        # incorrect = incorrect[["docId"]]
        # passages_incorrect = passages[passages["docId"].isin(incorrect["docId"])].copy()
        # passages_notincorrect = passages[~passages["docId"].isin(incorrect["docId"])]
        # passages_incorrect.drop_duplicates(subset=["passage"], keep='first', inplace=True)  # queda 233 pasaxes

        # add column with the class "correct"
        # passages_correct["labels"] = "correct"
        # passages_incorrect["labels"] = "incorrect"
        # merge both dataframes and remove column docId
        # pd_dataset = pd.concat([passages_correct, passages_incorrect], ignore_index=True)[["passage", "labels"]]
        # get rows of dataset of duplicated passages
        # duplicated_passages = pd_dataset[pd_dataset.duplicated(subset=["passage"])]
        descriptions = TrecDataLoader.get_topics('evaluation/misinfo-resources-2020/topics/misinfo-2020-topics.xml')
        passages['topic_description'] = passages.topic.apply(lambda x: descriptions[int(x)])
        return passages

    @staticmethod
    def get_topics(topics_file: str = 'evaluation/misinfo-resources-2021/topics/misinfo-2021-topics.xml'):
        import xml.etree.ElementTree as ET
        root = ET.parse(topics_file).getroot()
        topics = {}
        for topic in root.findall('topic'):
            description = topic.find("description").text
            number = topic.find("number").text
            # stance = topic.find("stance").text
            topics[int(number)] = description
        return topics
