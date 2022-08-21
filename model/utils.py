import numpy as np
import pandas as pd
import os, pickle
import sklearn.preprocessing

class ClassEncoder:
    def __init__(self, class_ids_path):
        self.class_ids_path = os.path.normpath(class_ids_path)
        if os.path.splitext(self.class_ids_path)[-1] == ".csv":
            self.classes = pd.read_csv(self.class_ids_path, index_col="id")
        else:
            with open(class_ids_path, 'rb') as handle:
                self.classes = list(pickle.load(handle))[0]
                self.classes = pd.DataFrame().assign(reference_id = self.classes)
        self.number_classes = self.classes.shape[0]
        self.enc = sklearn.preprocessing.OneHotEncoder()
        self.enc.fit(self.classes)

    def encode(self, target_column):
        return self.enc.transform(np.array(target_column).reshape(-1, 1))

    def decode_one(self, class_id):
        if class_id is None:
            return None
        else:
            return self.classes.loc[class_id, "reference_id"]

    def decipher(self, classes):
        # returns list
        return list(map(lambda class_id: self.decode_one(class_id), classes))

class DataLoader:
    def __init__(self, class_ids_path):
        #self.csv_path = os.path.normpath(csv_path)
        self.number_features = 0
        self.class_encoder = ClassEncoder(class_ids_path)

    def load(self, csv_path):
        self.X = pd.read_csv(csv_path)
        self.number_features = self.X.shape[1] - 1
        self.y = self.X["reference_id"]
        self.X.drop(columns=["reference_id"], inplace=True)
        self.y = self.class_encoder(self.y)

def make_prediction(preprocessor, model, class_encoder, products):
    preprocessed_data = preprocessor.preprocess(products)
    model_predictions = model.predict(preprocessed_data)  # class ids & None if not found
    return class_encoder.decipher(model_predictions)

if __name__ == "__main__":
    data_loader = DataLoader(csv_path="..//data//products//product_data.csv",
                             class_ids_path = "..//data//products//classes.csv")
    data_loader.load()