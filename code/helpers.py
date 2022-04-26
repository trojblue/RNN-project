
# === tokenizer ===
import numpy as np
import re, torch, time
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import f1_score, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_word(posts):
    tknzr = TweetTokenizer()
    tokenized_posts = [tknzr.tokenize(post) for post in posts]
    return tokenized_posts

def remove_punctuation(posts):
    posts_without_punctuation = [[word if len(word) == 2 else re.sub(r'[^\w\s]', '', word) for word in post] for post in
                                 posts]
    posts_without_punctuation = [[word for word in post if not len(word) == 0] for post in
                                 posts_without_punctuation]  # remove space
    return posts_without_punctuation


# 用不同的数据预处理组合来处理数据
def data_preprocess(posts):
    posts_tokenisation = tokenize_word(posts)
    posts_puncuation_removal = remove_punctuation(posts_tokenisation)
    return " ".join(posts_puncuation_removal[0])


from nltk import word_tokenize


def my_tokenizer(text):
    return word_tokenize(data_preprocess([text]))


def train_fuc(train_iter, test_iter, net, loss, optimizer, num_epochs):
    """training the model
    """
    batch_count, test_f1, train_f1, train_acc, test_accuracy = 0, [], [], [], []

    for epoch in range(num_epochs):
        preds = []
        labels = []
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        allLoss = 0.
        for batch_idx, batch in enumerate(train_iter):
            post_type, text1 = batch.label.to(DEVICE), batch.text.to(DEVICE)

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
            optimizer.zero_grad()  # 将优化器的梯度归为0
            ls.backward()  # 损失回传，计算梯度
            optimizer.step()  # 优化器优化模型参数
            train_l_sum += ls.item()  # 计算训练损失
            allLoss += ls.item()
            train_acc_sum += (y_hat.argmax(dim=1) == post_type).sum().item()
            n = n + post_type.shape[0]
            batch_count += 1

        preds = np.array(preds)
        labels = np.array(labels)
        train_f1score = f1_score(labels, preds)

        test_acc, f1 = evaluation_fuc(test_iter, net)

        test_f1.append(f1)
        train_f1.append(train_f1score)
        train_acc.append(train_acc_sum / n)
        test_accuracy.append(test_acc)

        print(
            'Epoch %d, train accuracy %.3f, test accuracy %.3f, test f1 %.3f'
            % (epoch + 1, train_acc_sum / n, test_acc, f1
               ))

    return train_f1, test_f1, train_acc, test_accuracy

def evaluation_fuc(test_iter, model):
    acc_sum, n = 0.0, 0
    preds = []
    labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):  # 获取训练批次
            post_type, text1 = batch.label.to(DEVICE), batch.text.to(DEVICE)  # 每个训练批次里面有标签和文本
            post_type.data.sub_(1)  # 由于标签中会自动有一个‘unk’的标签占位，而这个是不需要训练的，所以需要把对应标签数量减一
            model.eval()  # 将模型设置为测试模式
            prediction = model(text1).argmax(dim=1)  # 获取模型预测概率，并选取概率最大的作为模型的预测结果
            accuracy_num = (prediction == post_type).float().sum().item()  # 计算准确率
            acc_sum = acc_sum + accuracy_num
            label = post_type.cpu().numpy().tolist()  # 记录所有真实标签
            pred = prediction.cpu().numpy().tolist()  # 记录预测结果
            for l, p in zip(label, pred):
                if l != -1:
                    preds.append(p)
                    labels.append(l)
            model.train()
            n = n + post_type.shape[0]
    preds = np.array(preds)
    labels = np.array(labels)
    f1score = f1_score(labels, preds)  # 计算F1值

    print(classification_report(labels, preds, digits=4))  # 答应所有细致的分类结果
    # print(f'test acc is {accuracy_score(labels,preds)}')
    return acc_sum / n, f1score


if __name__ == '__main__':
    # 初次运行, 拆分csv前

    import nltk
    nltk.download('stopwords')