import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import re
from transformers import AutoModel, AutoTokenizer, BertTokenizer
import torch

class MyDataset(Dataset):
    def __init__(self, data_path, dict_path, max_length_sentences=70, max_length_word=70,idx_label = 1,is_processed_data = False,bert_model = ""):
        super(MyDataset, self).__init__()
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load("./classes.npy")
        if bert_model != "":
            self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        else:
            #Tạo từ điển Embedding
            dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE, usecols=[0]).values
            dict = [word[0] for word in dict]
            self.new_dict = {}
            for idx,word in enumerate(dict,start=1):
                self.new_dict[word] = idx

        if not is_processed_data:
            self.texts, self.labels = self.read_file_json(data_path,idx_label)
            self.documents_encode = self.process(bert_model=bert_model)
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

    def read_file_json(self,json_file,idx_label):
        texts, labels = [], []
        with open(json_file, encoding='utf-8') as json_file:
            dataset = json.load(json_file)
            for record in dataset[:500]:
                title = record.get("title","")
                description = record.get("description","")
                content = record.get("content","")
                label = record.get("label",[0,0])
                text = title+"\n"+description+"\n"+content
                texts.append(text)
                label = self.label_encoder.transform(label)
                label = label[idx_label]
                if idx_label == 1:
                    label = label-3
                labels.append(label)
        return texts, labels            
    
    def get_sentence_embedding_from_bert(self,sentence):
        return self.tokenizer.encode(sentence, max_length=self.max_length_word,padding="max_length")
        
    def get_sentence_embedding(self, sentence):
        sentence_encode = []
        for word in word_tokenize(text=sentence):
            word_encode = self.new_dict.get(word.lower(),-1)
            sentence_encode.append(word_encode)
        if len(sentence_encode) < self.max_length_word:
            extended_words = [-1 for _ in range(self.max_length_word - len(sentence_encode))]
            sentence_encode.extend(extended_words)
        return sentence_encode

    def preprocessing_text(self,text):
        text = re.sub("\r","\n",text)
        text = re.sub("\n{2,}","\n",text)
        text = re.sub("…",".",text)
        text = re.sub("\.{2,}",".",text)
        text.strip()
        return text
    
    def process(self,bert_model):
        documents_encode = []
        for i in tqdm(range(0,len(self.texts))):
            # Thêm tách câu bằng dấu \n nữa
            text = self.texts[i]
            text = self.preprocessing_text(text)
            paragraphs = text.split("\n")
            document_encode = []
            for paragraph in paragraphs:
                for sentence in sent_tokenize(text=paragraph):
                    if bert_model != "":
                        sentence_encode = self.get_sentence_embedding_from_bert(sentence=sentence)
                    else:
                        sentence_encode = self.get_sentence_embedding(sentence=sentence)
                    document_encode.append(sentence_encode)
            document_encode = self.padding_data(document_encode)
            documents_encode.append(document_encode)
        return documents_encode

    #Thêm padding cho các đoạn văn
    def padding_data(self,document_encode):
        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[0 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)

        return document_encode.astype(np.int64)

    def __getitem__(self, index):
        label = self.labels[index]
        document_encode = self.documents_encode[index]
        return document_encode,label

if __name__ == '__main__':
    train = MyDataset(data_path="./dataset/vosint/train.json", dict_path="./models/glove.6B.300d.txt",is_processed_data=False)
    test = MyDataset(data_path="./dataset/vosint/test.json",
    dict_path="./models/glove.6B.300d.txt",is_processed_data=False)
