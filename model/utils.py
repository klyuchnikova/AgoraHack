import numpy as np
import pandas as pd
import os
import sklearn.preprocessing

class DataLoader:
    def __init__(self, csv_path, class_ids_path):
        self.csv_path = os.path.normpath(csv_path)
        self.class_ids_path = os.path.normpath(class_ids_path)
        self.number_classes = 0
        self.number_features = 0

    def load(self):
        self.data = pd.read_csv(self.csv_path)
        self.classes = pd.read_csv(self.class_ids_path).reindex(["reference_id", "id"]).to_numpy()
        self.number_classes = self.classes.shape[0]
        self.number_features = self.data.shape[1] - 1
        self.enc = sklearn.preprocessing.OneHotEncoder()
        self.enc.fit(self.classes)
        predicts = self.enc.transform(data["reference_id"])
        print(predicts)
        #return data, enc.categories_()

    def id_to_class(self, class_id):
        return self.classes.loc[class_id, "reference_id"]

def make_prediction(preprocessor, model, loader, products):
    preprocessed_data = preprocessor.preprocess(products)
    model_predictions = model.predict(preprocessed_data)  # class ids & None if not found
    np.apply_along_axis(loader.id_to_class, model_predictions[model_predictions != None])
    return model_predictions

import re
from torchtext.vocab import build_vocab_from_iterator
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

def split_texts(dataset):
    for text in dataset:
        yield re.sub('[^A-Za-z ]+', '', text).split()

def create_vocab(data):
    vocab = build_vocab_from_iterator(split_texts(data["props"]), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

if __name__ == "__main__":
    data = pd.read_csv("..//data//products//cleaned_data.csv")
    data.drop(columns=["Unnamed: 0"], inplace=True)
    vocab = create_vocab(data)
    text_pipeline = lambda x: vocab(x.split())
    label_pipeline = lambda x: int(x) - 1
    vectorizer = CountVectorizer(vocabulary=vocab.get_itos())
    product_data = pd.read_csv("..//data//products//product_data.csv")
    print(product_data.columns)
    product_data["props"] = vectorizer.transform(product_data["props"]).todense()
    product_data.drop("is_reference", inplace=True)
    #vocab = create_vocab()