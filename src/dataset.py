import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import re

class MyDataset(Dataset):
    def __init__(self, data_path, dict_path, max_length_sentences=70, max_length_word=50,idx_label = 1,is_processed_data = False):
        super(MyDataset, self).__init__()
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load("./classes.npy")
        if not is_processed_data:
            texts, labels = [], []
            with open(data_path) as json_file:
                dataset = json.load(json_file)
                for record in dataset:
                    title = record.get("title","")
                    description = record.get("description","")
                    content = record.get("content","")
                    label = record.get("label",[0,0])
                    text = title+"\n"+description+"\n"+content
                    texts.append(text)
                    label = label_encoder.transform(label)
                    label = label[idx_label]
                    if idx_label == 1:
                        label = label-3
                    labels.append(label)

            self.texts = texts
            self.labels = labels
            self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE, usecols=[0]).values
            self.dict = [word[0] for word in self.dict]
            self.new_dict = {}
            for idx,word in enumerate(self.dict):
                self.new_dict[word] = idx
            self.documents_encode = self.process()
            assert len(self.documents_encode) == len(self.labels)
            dataset = [self.documents_encode,self.labels]
            new_dataset = np.asarray(dataset,dtype=object)
            args_path = data_path.split("/")
            name_file = args_path[-1].split('.')[0]
            str_path = "/".join(args_path[:-1])
            np.save(str_path+'/'+name_file+'.npy',new_dataset)
        else:
            dataset = np.load(data_path,allow_pickle=True)
            self.documents_encode = dataset[0]
            self.labels = dataset[1]
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def preprocessing_text(self,text):
        text = re.sub("\r","\n",text)
        text = re.sub("\n{2,}","\n",text)
        text = re.sub("…",".",text)
        text = re.sub("\.{2,}",".",text)
        text.strip()
        return text
    
    def process(self):
        # word_set_un = set()
        documents_encode = []
        for i in tqdm(range(0,len(self.texts))):
            # Thêm tách câu bằng dấu \n nữa
            text = self.texts[i]
            text = self.preprocessing_text(text)
            paragraphs = text.split("\n")
            document_encode = []
            for paragraph in paragraphs:
                for sentence in sent_tokenize(text=paragraph):
                    sentence_encode = []
                    for word in word_tokenize(text=sentence):
                        word_encode = self.new_dict.get(word.lower(),-1)
                        # if word_encode == -1:
                        #     word_set_un.add(word)
                        sentence_encode.append(word_encode)
                    document_encode.append(sentence_encode)
            document_encode = self.padding_data(document_encode)
            documents_encode.append(document_encode)
        # print(word_set_un)
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
        label = self.labels[index]
        document_encode = self.documents_encode[index]
        return document_encode,label


if __name__ == '__main__':
    train = MyDataset(data_path="./dataset/vosint/train.json",
        dict_path="./models/glove.6B.300d.txt",
        is_processed_data=False)
    test = MyDataset(data_path="./dataset/vosint/test.json",
    dict_path="./models/glove.6B.300d.txt",
    is_processed_data=False)
