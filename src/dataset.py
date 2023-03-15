"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from pandas.io.parquet import doc
from pandas.io.pickle import pickle
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, data_path, dict_path, max_length_sentences=100, max_length_word=50,is_processed_data = False):
        super(MyDataset, self).__init__()
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        if not is_processed_data:
            texts, labels = [], []
            with open(data_path) as csv_file:
                reader = csv.reader(csv_file, quotechar='"')
                for idx, line in enumerate(reader):
                    text = ""
                    for tx in line[1:]:
                        text += tx.lower()
                        text += " "
                    label = int(line[0])
                    texts.append(text)
                    labels.append(label)

            self.texts = texts
            self.labels = labels
            self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                    usecols=[0]).values
            self.dict = [word[0] for word in self.dict]
            self.new_dict = {}
            for idx,word in enumerate(self.dict):
                self.new_dict[word] = idx
            self.documents_encode = self.process()
            assert len(self.documents_encode) == len(self.labels)
            dataset = [self.documents_encode,self.labels]
            new_dataset = np.asarray(dataset,dtype=object)
            args_path = data_path.split("/")
            print(args_path)
            name_file = args_path[-1].split('.')[0]
            print(name_file)
            str_path = "/".join(args_path[:-1])
            np.save(str_path+'/'+name_file+'.npy',new_dataset)
        else:
            dataset = np.load(data_path,allow_pickle=True)
            self.documents_encode = dataset[0]
            self.labels = dataset[1]

        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def process(self):
        documents_encode = []
        for i in tqdm(range(0,len(self.texts))):
            text = self.texts[i]
            document_encode = []
            for sentence in sent_tokenize(text=text):
                sentence_encode = []
                for word in word_tokenize(text=sentence):
                    word_encode = self.new_dict.get(word,-1)
                    sentence_encode.append(word_encode)
                document_encode.append(sentence_encode)
            document_encode = self.padding_data(document_encode)
            documents_encode.append(document_encode)
        return documents_encode

    def padding_data(self,document_encode):
        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1
        return document_encode.astype(np.int64)

    def __getitem__(self, index):
        # label = self.labels[index]
        # text = self.texts[index]
        # document_encode = [
        #     [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
        #     in
        #     sent_tokenize(text=text)]

        # for sentences in document_encode:
        #     if len(sentences) < self.max_length_word:
        #         extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
        #         sentences.extend(extended_words)

        # if len(document_encode) < self.max_length_sentences:
        #     extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
        #                           range(self.max_length_sentences - len(document_encode))]
        #     document_encode.extend(extended_sentences)

        # document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
        #                   :self.max_length_sentences]

        # document_encode = np.stack(arrays=document_encode, axis=0)
        # document_encode += 1

        # return document_encode.astype(np.int64), label

        label = self.labels[index]
        document_encode = self.documents_encode[index]
        return document_encode,label


if __name__ == '__main__':
    train = MyDataset(data_path="./dataset/plcx/train.csv",
     dict_path="./models/glove.6B.300d.txt",
     is_processed_data=False)
    test = MyDataset(data_path="./dataset/plcx/test.csv",
    dict_path="./models/glove.6B.300d.txt",
    is_processed_data=False)
