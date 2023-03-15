"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import get_evaluation
from src.dataset import MyDataset
import argparse
import shutil
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str, default="./dataset/plcx/test.csv")
    parser.add_argument("--pre_trained_model", type=str, default="./trained_models/plcx/best_model.pt")
    parser.add_argument("--word2vec_path", type=str, default="./models/glove.6B.300d.txt")
    parser.add_argument("--output", type=str, default="predictions/plcx")
    args = parser.parse_args()
    return args


def test(opt):
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)
    if torch.cuda.is_available():
        model = torch.load(opt.pre_trained_model)
    else:
        model = torch.load(opt.pre_trained_model, map_location=lambda storage, loc: storage)
    test_set = MyDataset(opt.data_path, opt.word2vec_path, model.max_sent_length, model.max_word_length)
    test_generator = DataLoader(test_set, **test_params)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    te_label_ls = []
    te_pred_ls = []
    for te_feature, te_label in test_generator:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature)
            te_predictions = F.softmax(te_predictions)
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())
    te_pred = torch.cat(te_pred_ls, 0).numpy()
    te_label = np.array(te_label_ls)
    encoder = LabelEncoder()
    encoder.classes_ = np.load("./dataset/plcd/classes.npy")
    fieldnames = ['True label', 'Predicted label', 'Content']
    with open(opt.output + os.sep + "predictions.csv", 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        
        for i, j, k in zip(te_label, te_pred, test_set.texts):
            true_labels = encoder.inverse_transform([i+3])[0]
            pred_labels = encoder.inverse_transform([np.argmax(j)+3])[0]
            writer.writerow(
                {'True label': true_labels, 'Predicted label': pred_labels, 'Content': k})

    test_metrics = get_evaluation(te_label, te_pred,
                                  list_metrics=["accuracy", "loss", "confusion_matrix"])
    print("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                               test_metrics["accuracy"],
                                                                               test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    opt = get_args()
    test(opt)
