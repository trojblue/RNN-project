# === tokenizer ===
import re
import time
import torch

import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import f1_score, classification_report
from nltk import word_tokenize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_word(reviews):
    """tokenize reviews using tknzr tokenizer
    """
    tknzr = TweetTokenizer()
    tokenized_posts = [tknzr.tokenize(post) for post in reviews]
    return tokenized_posts


def keep_text(reviews):
    """remove punctuations like <>,./? from reviews;
    only keep the words and white space
    """
    posts_without_punctuation = [
        [word if len(word) == 2 else re.sub(r'[^\w\s]', '', word) for word in post]
                for post in reviews]

    posts_without_punctuation = [[word for word in post if not len(word) == 0]
                 for post in posts_without_punctuation]  # remove space

    return posts_without_punctuation


def data_preprocess(reviews):
    """preprocess data based on the rules
    """
    posts_tokenisation = tokenize_word(reviews)
    posts_puncuation_removal = keep_text(posts_tokenisation)
    return " ".join(posts_puncuation_removal[0])


def my_tokenizer(text):
    return word_tokenize(data_preprocess([text]))


def evaluation_fuc(test_iter, model):
    """evaluating the model
    """
    acc_sum, n = 0.0, 0
    preds = []
    labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):  # loop through batches
            post_type, text1 = batch.label.to(DEVICE), batch.text.to(DEVICE)

            # labels contain placeholder '<unk>', reduce by one to avoid training on it
            post_type.data.sub_(1)
            model.eval()

            # making predictions based on likelihood
            prediction = model(text1).argmax(dim=1)

            # record accuracy, actual labels and prediction results
            accuracy_num = (prediction == post_type).float().sum().item()
            acc_sum = acc_sum + accuracy_num
            label = post_type.cpu().numpy().tolist()
            pred = prediction.cpu().numpy().tolist()
            for l, p in zip(label, pred):
                if l != -1:
                    preds.append(p)
                    labels.append(l)
            model.train()
            n = n + post_type.shape[0]

    # print report
    preds = np.array(preds)
    labels = np.array(labels)
    f1score = f1_score(labels, preds)
    print(classification_report(labels, preds, digits=4))

    # returns accuracy and f1 score
    return acc_sum / n, f1score


def train_fuc(train_iter, test_iter, net, loss, optimizer, num_epochs):
    """training the model

    """
    batch_count, test_f1, train_f1, train_acc, test_accuracy = 0, [], [], [], []
    loss_list = []

    for epoch in range(num_epochs):
        preds = []
        labels = []
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        allLoss = 0.

        for batch_idx, batch in enumerate(train_iter): # loop through batches
            post_type, text1 = batch.label.to(DEVICE), batch.text.to(DEVICE)

            # labels contain placeholder '<unk>', reduce by one to avoid training on it
            post_type.data.sub_(1)

            y_hat = net(text1)
            ls = loss(y_hat, post_type)
            pd = y_hat.argmax(dim=1)
            label = post_type.cpu().numpy().tolist()
            pred = pd.cpu().numpy().tolist()
            for l, p in zip(label, pred):
                if l != -1:
                    preds.append(p)
                    labels.append(l)

            # backward steps
            optimizer.zero_grad()  # set optimizer gradient to zero
            ls.backward()  # calculate gradient based on loss
            optimizer.step()  # optimizing model parameters

            # record results
            train_l_sum += ls.item()
            allLoss += ls.item()
            train_acc_sum += (y_hat.argmax(dim=1) == post_type).sum().item()
            n = n + post_type.shape[0]
            batch_count += 1

        # record results
        preds = np.array(preds)
        labels = np.array(labels)
        train_f1score = f1_score(labels, preds)
        test_acc, f1 = evaluation_fuc(test_iter, net)

        test_f1.append(f1)
        train_f1.append(train_f1score)
        train_acc.append(train_acc_sum / n)
        test_accuracy.append(test_acc)
        loss_list.append(train_l_sum / n )

        print(
            'Epoch %d, train accuracy %.3f, test accuracy %.3f, test f1 %.3f'
            % (epoch + 1, train_acc_sum / n, test_acc, f1) )

    return train_f1, test_f1, train_acc, test_accuracy, loss_list


if __name__ == '__main__':

    print("This is a helper file. run RNN.py instead")
