"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.hierarchical_att_model import HierAttNet
from torch.utils.tensorboard import SummaryWriter
import argparse
import shutil
import numpy as np
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoches", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=10,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="./dataset/plcx/train.npy")
    parser.add_argument("--test_set", type=str, default="./dataset/plcx/test.npy")
    parser.add_argument("--test_interval", type=int, default=5, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="./models/glove.6B.300d.npy")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models/plcx")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": True,
                   "drop_last": False}

    # max_word_length, max_sent_length = get_max_lengths(opt.train_set)
    max_word_length, max_sent_length = 30,35
    print(max_word_length, max_sent_length)
    print("=== Load Train Dataset ===")
    training_set = MyDataset(opt.train_set, opt.word2vec_path, max_sent_length, max_word_length,True)
    training_generator = DataLoader(training_set, pin_memory=True, **training_params)
    print("=== Load Test Dataset ===")
    test_set = MyDataset(opt.test_set, opt.word2vec_path, max_sent_length, max_word_length,True)
    test_generator = DataLoader(test_set, **test_params)
    # test_set = training_set
    # test_generator = training_generator
    print("=== Init Model ===")
    # training_set.num_classes = 10
    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes,
                       opt.word2vec_path, max_sent_length, max_word_length)

    print(model)
    print("=== Init Model  Done ===")

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_loss = 1e5
    best_epoch = 0
    best_acc = 0
    model.train()
    best_model = None
    number_not_good_epoch = 0
    print("="*10+"Train"+"+"*10)
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        for iter, (feature, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
            if iter % opt.test_interval == 0:
        # if epoch % opt.test_interval == 0:
                model.eval()
                loss_ls = []
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
                    te_loss = criterion(te_predictions, te_label)
                    loss_ls.append(te_loss * num_sample)
                    te_label_ls.extend(te_label.clone().cpu())
                    te_pred_ls.append(te_predictions.clone().cpu())
                te_loss = sum(loss_ls) / test_set.__len__()
                te_pred = torch.cat(te_pred_ls, 0)
                te_label = np.array(te_label_ls)
                test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy","loss", "confusion_matrix"])
                output_file.write(
                    "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n Test F1-Score: \n{}\n".format(
                        epoch + 1, opt.num_epoches,
                        te_loss,
                        test_metrics["accuracy"],
                        test_metrics["confusion_matrix"],
                        test_metrics['F1']))
                print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    optimizer.param_groups[0]['lr'],
                    te_loss, test_metrics["accuracy"]))
                writer.add_scalar('Test/Loss', te_loss, epoch)
                writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
                model.train()
                
                if best_acc < test_metrics['accuracy']:
                    best_acc = test_metrics['accuracy']
                    print("Best model is {}",test_metrics["accuracy"])
                    torch.save(model, opt.saved_path + os.sep + "best_model.pt")
                    number_not_good_epoch = 0
                # if te_loss + opt.es_min_delta < best_loss:
                #     best_loss = te_loss
                #     best_epoch = epoch
                #     print("Best model is {}",test_metrics["accuracy"])
                #     torch.save(model, opt.saved_path + os.sep + "whole_model_han")
        number_not_good_epoch += 1
            # Early stopping
            # if epoch - best_epoch > opt.es_patience > 0:
        if number_not_good_epoch > opt.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
            break


if __name__ == "__main__":
    opt = get_args()
    train(opt)