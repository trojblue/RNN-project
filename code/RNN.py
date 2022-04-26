import matplotlib.pyplot as plt
from torchtext import data
from torchtext.vocab import Vectors

from helpers import *
from rnnmodel import *


def setup():
    """setup files & dependencies for the project
    """
    import pandas as pd

    # read input data
    all_data = pd.read_csv("../data/data.csv")
    df_empty = pd.DataFrame(columns=['label', 'text'])
    df_empty['label'] = all_data['Recommended IND']
    df_empty['text'] = all_data['ReviewText']

    # separate train, val and test
    lens = len(df_empty)
    train_data = df_empty[0:int(lens * 0.7)]
    val_data = df_empty[int(lens * 0.7):int(lens * 0.9)]
    test_data = df_empty[int(lens * 0.9):]

    train_data.to_csv("../data/train_data.csv", index=False, sep=',')
    val_data.to_csv("../data/val_data.csv", index=False, sep=',')
    test_data.to_csv("../data/test_data.csv", index=False, sep=',')

    # install required nltk pack
    import nltk
    nltk.download('stopwords')


def run():
    text_len = 60

    # defining text attributes
    text = data.Field(sequential=True,
                      batch_first=True,
                      fix_length=text_len,
                      tokenize=my_tokenizer)

    label = data.Field(sequential=False)

    # reading csv files
    train, val, test = data.TabularDataset.splits(
        path='../data/',
        train='train_data.csv',
        validation='val_data.csv',
        test='test_data.csv',
        skip_header=True,
        format='csv',
        fields=[('label', label), ('text', text)])

    # using pretrained glove embedding
    vectors = Vectors("../data/glove.6B.50d.txt")

    # building vocabulary
    text.build_vocab(train, val, test,
                     vectors=vectors,
                     unk_init=torch.Tensor.normal_)

    label.build_vocab(train, val)
    pretrained_ebd = text.vocab.vectors

    # hyper-parameters
    batch_size = 64
    ebdsize = 50  # glove embedding used has a size of 50
    num_hiddens = 64  # set smaller to reduce over-fitting
    dropout_rate = 0.5  # 50% dropout
    vocab_size = len(text.vocab)
    output_dim = 2  # 'recommend' or 'not recommend'
    label_num = 2

    # cut training set into batches
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test),
        sort_key=lambda x: len(x.text),
        batch_sizes=(batch_size, len(val), len(test))
    )

    # initializing model

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BIRNN(vocab_size, ebdsize, num_hiddens, pretrained_ebd, dropout=dropout_rate)
    model = model.to(DEVICE)
    print(model)

    # define training attributes
    lr, num_epochs = 0.001, 10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # training (training & evaluation functions are defined in helpers.py)
    print("training model....")
    train_f1, test_f1, train_acc, test_accuracy, loss_list = \
        train_fuc(train_iter, val_iter, model, loss, optimizer, num_epochs)

    t_acc, t_f1 = evaluation_fuc(test_iter, model)
    print(f'test acc is {t_acc},test f1 is {t_f1}')

    # drawing plots
    plot_graphs(num_epochs, test_accuracy, test_f1, train_acc, train_f1, loss_list)


def plot_graphs(num_epochs, test_accuracy, test_f1, train_acc, train_f1, loss_list):
    """drawing graphs for qualitative & quantitative measures
    """
    ep = list(range(num_epochs))
    plt.plot(ep, loss_list, 'r--', label='loss')  # loss over epoch
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("../data/loss.jpg")
    plt.clf()



    ep = list(range(num_epochs))
    plt.plot(ep, train_acc, 'r--', label='train_accuracy')  # test/train accuracy
    plt.plot(ep, test_accuracy, 'g--', label='test_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("../data/accuracy.jpg")
    plt.clf()

    l1 = plt.plot(ep, train_f1, 'r--', label='train_f1')  # test/train f1 score
    l2 = plt.plot(ep, test_f1, 'g--', label='test_f1')
    plt.xlabel('Epoch')
    plt.ylabel('F1_score')
    plt.legend()
    plt.savefig("../data/f1_score.jpg")
    plt.clf()


if __name__ == '__main__':
    # setup()
    run()
