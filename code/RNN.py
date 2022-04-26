from helpers import *

def setup():
    import pandas as pd

    all_data = pd.read_csv("../data/data.csv")
    df_empty = pd.DataFrame(columns=['label', 'text'])
    df_empty['label']=all_data['Recommended IND']
    df_empty['text']=all_data['ReviewText']

    lens = len(df_empty)

    train_data = df_empty[0:int(lens*0.7)]
    val_data = df_empty[int(lens*0.7):int(lens*0.9)]
    test_data = df_empty[int(lens*0.9):]

    train_data.to_csv("../data/train_data.csv",index=False,sep=',')
    val_data.to_csv("../data/val_data.csv",index=False,sep=',')
    test_data.to_csv("../data/test_data.csv",index=False,sep=',')

    import nltk
    nltk.download('stopwords')


def all():
    # import pandas as pd
    # import torch
    # import jieba
    # import os
    # from torch.nn import init
    # from torchtext import data
    from torchtext.vocab import Vectors
    import time
    # import math
    from rnnmodel import myRNN

    # import nltk
    from nltk.tokenize import TweetTokenizer
    import re




    # def covert_to_lowercase(posts):
    #   lower_posts = [[word.lower() for word in post] for post in posts]
    #   return lower_posts
    # from nltk.corpus import stopwords as sw
    # stop_words = sw.words()
    # def remove_stopwords(posts):
    #   posts_without_stopword = [[word for word in post if not word in stop_words] for post in posts]
    #   return posts_without_stopword

    import torch
    import torch.nn as nn
    from torchtext import data  # uses torchtext 0.6.0
    # import torch.nn.functional as F
    # import pandas as pd
    # from torchtext.legacy.data import data

    text_len=60

    # 定义文本格式，以便读取
    text = data.Field(sequential=True,
                    batch_first =True,
                    fix_length=text_len,
                    tokenize=my_tokenizer)

    label = data.Field(sequential=False)

    train, val,test = data.TabularDataset.splits(
        path = '../data/',
        train='train_data.csv',
        validation='val_data.csv',
        test='test_data.csv',
        skip_header=True,
        format='csv',
        fields=[('label', label), ('text', text)])


    vectors = Vectors("../data/glove.6B.50d.txt")

    text.build_vocab(train, val, test,
                    vectors=vectors,
                    unk_init = torch.Tensor.normal_)

    label.build_vocab(train, val)
    pretrained_ebd = text.vocab.vectors



    class BIRNN(nn.Module):
        def __init__(self,vocab_size,ebdsize,num_hiddens):
            super(BIRNN, self).__init__()
            self.emb = nn.Embedding(vocab_size, ebdsize)
            self.emb = self.emb.from_pretrained(pretrained_ebd, freeze=False)
            self.RNN = myRNN(50,
            hidden_size=num_hiddens,
            batch_first=True,
            bidirectional=True)
            self.linear = nn.Linear(num_hiddens*2, 2)

        def forward(self, text1):
            emb =  self.emb(text1)
            o_n, h_n,= self.RNN(emb)
            hidden_out = torch.cat((h_n[0,:,:],h_n[1,:,:]),1)
            out = self.linear(hidden_out)
            return out


    batch_size=64
    # 将训练数据切分batch
    train_iter, val_iter,test_iter = data.Iterator.splits(
                (train, val, test),
                sort_key=lambda x: len(x.text),
                batch_sizes=(batch_size, len(val),len(test))
        )

    # hyperparameters
    ebdsize = 50
    num_hiddens = 64
    dropout = 0.5  # 50% dropout
    vocab_size = len(text.vocab)
    output_dim = 2  # 'recommend' or 'not recommend'
    label_num = 2


    import numpy as np
    from sklearn.metrics import accuracy_score,f1_score,classification_report

    # 定义测试函数
    def evaluation_fuc(test_iter, model):
        acc_sum, n = 0.0, 0
        preds = []
        labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_iter):
                post_type, text1= batch.label.to(DEVICE), batch.text.to(DEVICE)
                post_type.data.sub_(1)
                model.eval()
                prediction = model(text1).argmax(dim=1)
                accuracy_num = (prediction== post_type).float().sum().item()
                acc_sum = acc_sum + accuracy_num
                label = post_type.cpu().numpy().tolist()
                pred = prediction.cpu().numpy().tolist()
                for l,p in zip(label, pred):
                    if l != -1:
                        preds.append(p)
                        labels.append(l)
                model.train()
                n = n + post_type.shape[0]
        preds = np.array(preds)
        labels = np.array(labels)
        f1score = f1_score(labels,preds)

        print(classification_report(labels,preds,digits=4))
        # print(f'test acc is {accuracy_score(labels,preds)}')
        return acc_sum / n, f1score

    # 定义训练函数
    def train_fuc(train_iter, test_iter, net, loss, optimizer, num_epochs):
        batch_count = 0
        test_f1 = []
        train_f1 = []
        train_acc =[]
        test_accuracy =[]
        for epoch in range(num_epochs):
            preds = []
            labels = []
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            allLoss=0.
            for batch_idx, batch in enumerate(train_iter):
                post_type, text1= batch.label.to(DEVICE), batch.text.to(DEVICE)

                post_type.data.sub_(1)
                y_hat = net(text1)
                ls = loss(y_hat, post_type)
                pd = y_hat.argmax(dim=1)
                label = post_type.cpu().numpy().tolist()
                pred = pd.cpu().numpy().tolist()
                for l,p in zip(label, pred):
                    if l != -1:
                        preds.append(p)
                        labels.append(l)
                optimizer.zero_grad()
                ls.backward()
                optimizer.step()
                train_l_sum += ls.item()
                allLoss+= ls.item()
                train_acc_sum += (y_hat.argmax(dim=1) == post_type).sum().item()
                n = n + post_type.shape[0]
                batch_count += 1

            preds = np.array(preds)
            labels = np.array(labels)
            train_f1score = f1_score(labels,preds)

            if epoch == num_epochs-1:
                test_acc,f1 = evaluation_fuc(test_iter, net)
            else:
                test_acc,f1 = evaluation_fuc(test_iter, net)
            test_f1.append(f1)
            train_f1.append(train_f1score)
            train_acc.append(train_acc_sum / n)
            test_accuracy.append(test_acc)

            print(
                'Epoch %d, train accuracy %.3f, test accuracy %.3f, test f1 %.3f'
                % (epoch + 1,  train_acc_sum / n, test_acc,  f1
                   ))

        return train_f1, test_f1, train_acc,test_accuracy

    # 初始化模型
    model = BIRNN(vocab_size,ebdsize,num_hiddens)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    print(model)
    # 定义训练参数
    lr, num_epochs =0.001, 10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # 训练
    print("training model....")
    train_f1,test_f1,train_acc,test_accuracy = train_fuc(train_iter, val_iter, model, loss, optimizer, num_epochs)

    t_acc, t_f1 =evaluation_fuc(test_iter, model)
    print(f'test acc is {t_acc},test f1 is {t_f1}')

    import numpy as np
    import matplotlib.pyplot as plt

    ep = list(range(num_epochs))

    # plt.plot(ep,train_acc,'r--',label='train_accuracy')
    # plt.plot(ep,test_accuracy,'g--',label='test_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig("fig2.jpg")

    l1=plt.plot(ep,train_f1,'r--',label='train_f1')
    l2=plt.plot(ep,test_f1,'g--',label='test_f1')
    plt.xlabel('Epoch')
    plt.ylabel('F1_score')
    plt.legend()
    plt.savefig("fig.jpg")



if __name__ == '__main__':
    # setup()
    all()
