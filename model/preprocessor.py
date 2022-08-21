import csv
import json
import os
import re
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator


def basic_cleaning(s):
    ss = s
    ss = ss.replace("(", "")
    ss = ss.replace(")", "")
    ss = ss.replace(",", "")
    ss = ss.replace('"', "")
    ss = ss.replace(' /', " ")
    return ss.lower()


def cleaning_string(s):
    ss = s.lower()
    ss = ss.replace(r"[", "")
    ss = ss.replace(r"]", "")
    ss = ss.replace(r"\\t", " ")
    ss = ss.replace(r"\t", " ")
    ss = ss.replace(r"...", " ")
    ss = ss.replace("(", "")
    ss = ss.replace(")", "")

    ss = ss.replace(r"'", '')
    return str(ss)


def string_to_list(s):
    ss = s
    ss = ss.replace("[", "")
    ss = ss.replace("]", "")
    ss = ss.replace(r"\\t", " ")
    ss = ss.replace(r"\t", " ")
    ss = ss[ss.find("\'") + 1:]
    ss = ss[:ss.rfind("\'")]
    ans = [basic_cleaning(word) for word in re.split("[\'\"][,]\s*[\'\"]", ss)]
    return ans


class Preprocessor:
    DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, text_vectorizer = None):
        self.text_vectorizer = text_vectorizer

    def preprocess(self, products):
        """ [product = {
            "id": "some_id_3",
            "name": "Название товара3",
            "props": [...]
          }, ...]"""
        products = pd.json_normalize(products)
        products["props"] = str(products["props"])
        #products["props"] = np.array(re.sub("[^а-яА-Яa-zA-Z0-9]", " ", text).split() for text in products["props"].to_numpy())
        products["name"] = np.array(re.sub("[^а-яА-Яa-zA-Z0-9]", " ", text).split() for text in products["name"].to_numpy())
        if self.text_vectorizer:
            #products["props"] = self.text_vectorizer.vectorize_texts(products["props"])
            return self.text_vectorizer.vectorize_texts(products["name"])

    def dataset_to_csv(self, in_path, out_path):
        # output: data with columns, target column class with class numbers, array where [i] -> name of class number i
        in_path = os.path.join(self.DIR_PATH, os.path.normpath(in_path))
        out_path = os.path.join(self.DIR_PATH, os.path.normpath(out_path))
        data = None
        with open(in_path, "r") as f:
            data = json.load(f)
        keys = data[0].keys()
        a_file = open(out_path, "w")
        dict_writer = csv.DictWriter(a_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
        a_file.close()

    def preprocess_dataset(self, in_path, out_path):
        in_path = os.path.join(self.DIR_PATH, os.path.normpath(in_path))
        out_path = os.path.join(self.DIR_PATH, os.path.normpath(out_path))
        self.dataset_to_csv(in_path, out_path)
        return pd.read_csv(out_path)

    def correct_is_reference(self, data):
        for i in range(0, len(data)):
            if data.loc[i, 'is_reference'] == 1:
                data.loc[i, 'reference_id'] = data.loc[i, 'product_id']

    def clean_data(self, data):
        self.correct_is_reference(data)
        data['props'] = data['props'].apply(cleaning_string)
        return data

    def get_all_features(self, data):
        pass

    def encode_classes(self, data):
        return pd.DataFrame(data["reference_id"].dropna().unique())

    def transform_classes(self, data, out_class_path):
        out_class_path = os.path.join(self.DIR_PATH, os.path.normpath(out_class_path))
        classes_list = self.encode_classes(data)
        classes_list["id"] = np.arange(0, classes_list.shape[0])
        classes_list = classes_list.rename(columns={0: "reference_id"})
        classes_list.to_csv(out_class_path, index=False)
        classes_list = dict(classes_list.to_numpy().tolist())
        # data["reference_id"] = data["reference_id"].apply(lambda c: classes_list[c])

    def create_vectorizer(self):
        data = pd.read_csv("..//data//products//cleaned_data.csv")
        data.drop(columns=["Unnamed: 0"], inplace=True)
        vocab = create_vocab(data)
        self.text_pipeline = lambda x: vocab(x.split())
        self.label_pipeline = lambda x: int(x) - 1
        self.vectorizer = CountVectorizer(vocabulary=vocab.get_itos())

    def encode_props(self, data):
        self.create_vectorizer()
        data["props"] = self.vectorizer.transform(data["props"]).todense()


def split_texts(dataset):
    for text in dataset:
        yield text.split()  # re.sub('[^A-Za-z ]+', '', text).split()


def create_vocab(data):
    vocab = build_vocab_from_iterator(split_texts(data["props"]), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


if __name__ == "__main__":
    path_to_json = "data//products//agora_hack_products.json"
    out_path = "data//products//agora_hack_products.csv"
    p = Preprocessor()
    # data = p.preprocess_dataset(path_to_json, out_path)
    data = pd.read_csv("E:\E\Copy\PyCharm\AgoraHack\data\products\cleaned_data_time.csv", index_col="Unnamed: 0").dropna()
    # p.encode_props(data)
    # p.transform_classes(data, "data//products//classes_example.csv")
    print(data.head())
