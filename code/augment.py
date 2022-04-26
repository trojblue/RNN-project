"""
reference:
https://www.kaggle.com/code/mikiota/data-augmentation-csv-txt-using-back-translation/notebook
"""


import os
import nlpaug.augmenter.word as naw
import pandas as pd

from tqdm.auto import tqdm

def setup():
    """ run this if it's the first time using it
    """
    import nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def augment():
    """
    augment training data by changing synonyms,
    translate to German and translate then
    translate back to English again
    """
    train = pd.read_csv('../data/train_data.csv')
    train.head()

    # try getting a single synonym change
    # text_chunk = train.iloc[10]['text']
    # syn_aug = naw.SynonymAug(aug_src='wordnet')
    # text_chunk_aug_syn = syn_aug.augment(text_chunk)

    back_trans_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de',
        to_model_name='facebook/wmt19-de-en',
        device='cuda',
        batch_size=32
    )

    # back translation to a new csv, it will download ~2 Gb of model first
    # it takes around 2 days running on my computer to translate 4000 entries into csv.
    train_selected = train[0:4000]
    augmenter = back_trans_aug
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tqdm.pandas()   # progress bar
    train_selected["text"] = train_selected.progress_apply(
        lambda row: augmenter.augment(row["text"]), axis=1)

    # save to csv
    train_selected.to_csv("../data/augmented_train.csv", index=False)
    print("Done")


if __name__ == '__main__':
    # setup()
    augment()



