import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from sklearn import preprocessing
from bs4 import BeautifulSoup
from langdetect import detect
import xml.etree.ElementTree as ET
import nltk
nltk.download('stopwords')
nltk.download('punkt')

class ReadClass:
    def __init__(self, directory):
        self.directory = directory
        self.data = pd.read_csv('./pygaggle/CLEF_2018/CLEF_read_labels.txt')
        # model_name = 'castorini/monot5-base-med-msmarco'
        # tokenizer_name = 't5-base'
        self.reranker =  MonoT5('castorini/monot5-base-med-msmarco')
        root = ET.parse('./pygaggle/CLEF_2018/queries.txt').getroot()
        self.queries = {}
        for query in root.findall('query'):
            number = query.find("id").text.strip()
            text = query.find("en").text.strip()
            self.queries[number] = text
        print("!!!Queries:", self.queries)

    '''
    This function creates a sliding window over text
    '''
    def __window(self, iterable, n = 6, m = 3):
        if m == 0: # otherwise infinte loop
            raise ValueError("Parameter 'm' can't be 0")
        lst = list(iterable)
        i = 0

        if i + n < len(lst):
            while i + n < len(lst):
                yield ' '.join(lst[i:i + n])
                i += m

        else:
            yield ' '.join(lst)
    
    '''
    This function generates the most relevant passage for each document-query pair
    '''
    def gen_passages(self):
        passages = []
        
        for topic,docId,site in zip(self.data.topic, self.data.docId, self.data.site):
            path = self.directory + '/' + str(site) + '/' + str(docId)
            passage = ""
            print(path)
            try:
                with open(path) as f:
                    soup =  BeautifulSoup(f.read(), "html.parser")
                    for script in soup(["script", "style"]): # boilerplate removal
                        script.decompose() 
                    doc = soup.get_text()
                    lines = (line.strip() for line in doc.splitlines()) # break into lines and remove leading and trailing space on each
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) # break multi-headlines into a line each
                    doc = ' '.join(chunk for chunk in chunks if chunk) # drop blank lines
                    doc = doc.replace('\\n', ' ').replace('\\r', '').replace('\\t', '').replace('\\', '') # remove end-line character
                    try:
                        lang = detect(doc)
                    except:
                        lang = "error"
                    if lang=="en":
                        doc = nltk.sent_tokenize(doc)
                        doc_passages = self.__window(doc) # len = 6, stride = 3
                        texts = [Text(p.encode("utf-8", errors="replace").decode("utf-8"), None, 0) for p in doc_passages]
            
                        query = Query(self.queries[str(topic)])
                        reranked = self.reranker.rerank(query, texts)
                        reranked.sort(key=lambda x: x.score, reverse=True)
                        if reranked:
                            passage = reranked[0].text 
                            passage = self.queries[str(topic)] + '\n' + passage
            except Exception as e:
                  print(e)
            passages.append(passage)

        self.data["passages"] = passages
        self.data.to_csv('./pygaggle/results/CLEF_read_labels_pass.txt', sep=' ', header=True, index=False)

    def gen_train_data(self):
        df = pd.read_csv('./pygaggle/results/CLEF_read_labels_pass.txt', sep=' ')
        vectorizer = TfidfVectorizer(min_df=0.1, stop_words=nltk.corpus.stopwords.words('english'))
        vectorizer.fit(df.passages)
        pickle.dump(vectorizer, open("./pygaggle/results/vectorizer.pickle", "wb"))
        data_train = vectorizer.transform(df.passages)
        scaler = preprocessing.MinMaxScaler().fit(list(data_train))
        pickle.dump(scaler, open("./pygaggle/results/scaler.pickle", "wb"))
        print("LABELS", dict(df.label.value_counts()))
        return scaler.transform(list(data_train)), df.label

    def train_model(self, X, y):
        clf = MultinomialNB()
        print(X.shape,len(y))
        clf.fit(X,y)
        pickle.dump(clf, open("./pygaggle/results/read_model.pickle", "wb"))

if __name__ == "__main__":
    cl = ReadClass('./pygaggle/clef2018collection')
    # passages = cl.gen_passages()
    X,y = cl.gen_train_data()
    cl.train_model(X,y)