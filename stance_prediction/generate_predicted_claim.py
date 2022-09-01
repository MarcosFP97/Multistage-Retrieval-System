import xml.etree.ElementTree as ET

import nltk
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
from supar import Parser


def special_join(leaves):  # join words like [at,-,home,working] as "at-home working" (detokenizer)
    string = leaves[0]
    i = 0
    while i < len(leaves) - 1:
        if leaves[i + 1] == '-':
            string += "".join([leaves[i + 1], leaves[i + 2]])
            i += 2
            continue
        string = TreebankWordDetokenizer().detokenize([string, leaves[i + 1]])
        i += 1
    return string


def get_affirmative(parser, topic):
    tok_question = nltk.word_tokenize(topic)
    parsed_question = parser.predict(tok_question, verbose=False)[0].values[-2]  # parsed sentence 0,values[-2]=tree
    for subtree in parsed_question.subtrees():
        if tok_question[0] not in subtree.leaves() and subtree.label() in ['NP', 'VP']:
            subject = special_join(subtree.leaves())
            afirm = str(subject + " " + tok_question[0].lower() + topic.split(subject)[1]).rstrip()[:-1]
            break
    return afirm


def get_negative(parser, topic):
    tok_question = nltk.word_tokenize(topic)
    print(tok_question)
    parsed_question = parser.predict(tok_question, verbose=False)[0].values[-2]  # parsed sentence 0,values[-2]=tree
    for subtree in parsed_question.subtrees():
        if tok_question[0] not in subtree.leaves() and subtree.label() in ['NP', 'VP']:
            subject = special_join(subtree.leaves())
            neg = str(subject + " " + tok_question[0].lower() + ' not' + topic.split(subject)[1]).rstrip()[:-1]
            break
    return neg

def main_variants():
    parser = Parser.load('crf-con-roberta-en')
    root = ET.parse('../../evaluation/misinfo-2022-topics.xml').getroot()
    topics = {}
    for topic in root.findall('topic'):
        topics[topic.find("number").text] = topic.find("question").text

    df = pd.read_csv('citius.gpt3.csv', names=["topic", "prediction"])
    with open('claims.txt', 'w') as f:
        for number, prediction in zip(df.topic, df.prediction):
            if prediction=="yes":
                f.write(str(number)+','+get_affirmative(parser, topics[str(number)])+ '\n') 
            elif prediction=="no":
                f.write(str(number)+','+get_negative(parser, topics[str(number)])+ '\n') 
            else:
                f.write(str(number)+','+topics[str(number)]+ '\n') 
            

if __name__ == '__main__':
    #nltk.download('averaged_perceptron_tagger', quiet=True)
    main_variants()
