"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
from transformers import AutoModel

class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50, bert_model=None):
        super(WordAttNet, self).__init__()
        self.relu = nn.ReLU()
        #Load embedding from csv
        if bert_model is None:
            dict = np.load(word2vec_path,allow_pickle=True)
            dict_len, embed_size = dict.shape
            dict_len += 1
            unknown_word = np.zeros((1, embed_size))
            dict = torch.from_numpy(np.concatenate([dict,unknown_word], axis=0))
            self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        else:
            # Load embedding from bert
            model = AutoModel.from_pretrained(bert_model)
            embed_size = 768
            self.lookup = model.embeddings.word_embeddings
            del model

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        output = self.lookup(input)
        output = output[0]
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output)
        output = element_wise_mul(f_output,output.permute(1,0))
        output = self.relu(output)
        return output, h_output

if __name__ == "__main__":
    abc = WordAttNet(bert_model="NlpHUST/vibert4news-base-cased")
    