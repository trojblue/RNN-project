
# === tokenizer ===
import re
from nltk.tokenize import TweetTokenizer



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





if __name__ == '__main__':
    # 初次运行, 拆分csv前

    import nltk
    nltk.download('stopwords')