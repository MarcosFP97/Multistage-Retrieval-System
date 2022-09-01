import xml.etree.ElementTree as ET

import nltk
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


def get_affirmatives(topics):
    affirmatives = []
    parser = Parser.load('crf-con-roberta-en')
    for question in topics:
        tok_question = nltk.word_tokenize(question)
        parsed_question = parser.predict(tok_question, verbose=False)[0].values[-2]  # parsed sentence 0,values[-2]=tree
        for subtree in parsed_question.subtrees():
            if tok_question[0] not in subtree.leaves() and subtree.label() in ['NP', 'VP']:
                subject = special_join(subtree.leaves())
                afirm = str(subject + " " + tok_question[0].lower() + question.split(subject)[1]).rstrip()[:-1]
                affirmatives.append(afirm)
                break
    return affirmatives


def get_negatives(topics):
    affirmatives = []
    parser = Parser.load('crf-con-roberta-en')
    for question in topics:
        tok_question = nltk.word_tokenize(question)
        parsed_question = parser.predict(tok_question, verbose=False)[0].values[-2]  # parsed sentence 0,values[-2]=tree
        for subtree in parsed_question.subtrees():
            if tok_question[0] not in subtree.leaves() and subtree.label() in ['NP', 'VP']:
                subject = special_join(subtree.leaves())
                print(subject)
                afirm = str(subject + " " + tok_question[0].lower() + ' not' + question.split(subject)[1]).rstrip()[:-1]
                affirmatives.append(afirm)
                break
    return affirmatives

def main_variants():
    root = ET.parse('../../evaluation/misinfo-2022-topics.xml').getroot()
    topics = []
    for topic in root.findall('topic/question'):
        topics.append(topic.text)

    with open('topic_variants.txt', 'w') as f:
        for topic, affirm, neg in zip(topics, get_affirmatives(topics), get_negatives(topics)):
            f.write(topic+','+affirm+','+neg+'\n')

if __name__ == '__main__':
    #nltk.download('averaged_perceptron_tagger', quiet=True)
    main_variants()
