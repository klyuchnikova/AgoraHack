import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch.nn as nn
import os

class ProductModel(nn.Module):
    DIR_PATH = os.path.join(os.getcwd(), "../data/models/")
    def __init__(self, file_name, input_size, number_classes, threshold = 0.2):
        # weight_file must be simply name, path to dir is defined
        super().__init__()
        self.input_size = input_size
        self.output_size = number_classes
        self.weight_file = os.path.join(ProductModel.DIR_PATH, file_name)

    def load(self, file_path):
        # absolute path
        pass

    def save(self, file_path):
        # absolute path
        pass

    def forward(self, X):
        # input: np.ndarray (batch_size, input_size)
        # X is ALREADY vectorized and tokenized
        # output : vector (batch_size, number_classes)
        pass

    def predict(self, X):
        return self.forward(X)
