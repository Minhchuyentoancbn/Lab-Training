from collections import defaultdict
from os import listdir
from os.path import isfile
from spacy.lang.en import English
import numpy as np
import re
import spacy


MAX_SENTENCES = 50
MAX_WORDS_PER_SENTENCE = 50
unknown_ID = 0
padding_ID = 1

nlp = English()
nlp.add_pipe('sentencizer')

def gen_data():
    def collect_data_from(parent_path, newsgroup_list):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = parent_path + r'/' + newsgroup + r'/'

            files = [(filename, dir_path + filename)
                     for filename in listdir(dir_path)
                     if isfile(dir_path + filename)]
            files.sort()
            label = group_id
            print(f'Processing: {group_id}-{newsgroup}')

            for filename, filepath in files:
                with open(filepath) as f:
                    text = f.read().lower()
                    
                    doc = nlp(text)
                    sentence_lst = [str(sen).strip() for sen in doc.sents]
                    words = [re.split(r'\W+', sent) for sent in sentence_lst]
                    content = '<fff>'.join([' '.join(sen) for sen in words])
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)

        return data

    path = '../Session1/data/20news-bydate/'
    parts = [path + dir_name + r'/' for dir_name in listdir(path)
             if not isfile(path + dir_name)]
    
    train_path, test_path = (parts[0], parts[1]) if 'train' in parts[0] else (parts[1], parts[0])

    newsgroup_list = [newsgroup for newsgroup in listdir(train_path)]
    newsgroup_list.sort()

    train_data = collect_data_from(
        parent_path=train_path,
        newsgroup_list=newsgroup_list
    )

    test_data = collect_data_from(
        parent_path=test_path,
        newsgroup_list=newsgroup_list
    )

    with open('data/w2v/20news-train-raw.txt', 'w') as f:
        f.write('\n'.join(train_data))

    with open('data/w2v/20news-test-raw.txt', 'w') as f:
        f.write('\n'.join(test_data))


def encode_data(data_path, vocab_path):
    with open(vocab_path) as f:
        vocab = dict([(word, word_ID + 2)
                      for word_ID, word in enumerate(f.read().splitlines())])

    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2:2+MAX_SENTENCES])
                     for line in f.read().splitlines()]
        
    encoded_data = []

    for document in documents:
        label, doc_id, text = document # text contains all sentences of the document
        encoded_doc = []
        
        for sentence in text:
            words = sentence.split()[:MAX_WORDS_PER_SENTENCE] # list of words in each sentence
            sentence_length = len(words)

            encoded_sent = []

            for word in words:
                if word in vocab:
                    encoded_sent.append(str(vocab[word]))
                else:
                    encoded_sent.append(str(unknown_ID))
        
            if sentence_length < MAX_WORDS_PER_SENTENCE:
                num_padding = MAX_WORDS_PER_SENTENCE - sentence_length
                for _ in range(num_padding):
                    encoded_sent.append(str(padding_ID))
                    
            encoded_doc.append(' '.join(encoded_sent))
            
        if len(text) < MAX_SENTENCES:
            num_padding = MAX_SENTENCES - len(text)
            for _ in range(num_padding):
                encoded_doc.append(' '.join([str(padding_ID)] * MAX_WORDS_PER_SENTENCE))

        encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>' + '<fff>'.join(encoded_doc))

    dir_name = '/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(dir_name + '/' + file_name, 'w') as f:
        f.write('\n'.join(encoded_data))
