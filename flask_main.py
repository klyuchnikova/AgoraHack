from flask import Flask, request
from model.model import ProductModel, TextVectorizer
from model.preprocessor import Preprocessor
from model.utils import make_prediction
from model.utils import DataLoader
import time
import os

PROJECT_PATH = os.path.dirpath(os.path.abspath(__file__))

app = Flask(__name__)
text_vectorizer = TextVectorizer() # vocab?
loader = DataLoader(csv_path = os.path.join(PROJECT_PATH, "data//products//processed_data.csv"),
                    class_ids_path = os.path.join(PROJECT_PATH, "data//products//classes.csv"))
number_classes = loader.number_classes
number_features = loader.number_features
model = ProductModel("model_v0", number_classes, number_features, threshold = 0.2)
preprocessor = Preprocessor()

@app.route('/match_products', methods=['POST'])
def predict_product():
    print(f'{time.time()}: start matching products')
    predictions = make_prediction(preprocessor, model, loader, request.form)
    return [{"id" : product.get("id"), "reference_id": predictions[i]} for i,product in enumerate(request.form)]

@app.route('/api/train_new', methods=['GET'])
def train_new_mode():
    pass

if __name__ == '__main__':
    app.run(host='localhost', port='8100', debug=True)