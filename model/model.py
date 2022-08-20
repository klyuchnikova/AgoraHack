import os

import numpy as np
import torch.nn as nn


class ProductModel(nn.Module):
    DIR_PATH = os.path.join(os.getcwd(), "../data/models/")

    def __init__(self, file_name, input_size, number_classes, vocab_size, threshold=0.2):
        # weight_file must be simply name, path to dir is defined
        super(ProductModel, self).__init__()
        self.input_size = input_size
        self.output_size = number_classes
        self.weight_file = os.path.join(ProductModel.DIR_PATH, file_name)
        self.threshold = threshold

        embed_dim = 100
        self.embed_level = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.lin = nn.Linear(embed_dim, self.output_size)
        self.out = nn.Softmax()

    def init_weights(self):
        initrange = 0.5
        self.embed_level.weight.data.uniform_(-initrange, initrange)
        self.lin.weight.data.uniform_(-initrange, initrange)
        self.lin.bias.data.zero_()

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
        self.X =
        pass

    def predict(self, X):
        # output: (batch_size, 1) -> class id or None
        predictions = np.argmax(self.forward(X), axis=1)
        return np.where(predictions > self.threshold, predictions, None)


class TextVectorizer:
    def __init__(self):
        pass
