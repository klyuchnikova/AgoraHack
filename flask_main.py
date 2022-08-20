from flask import Flask, request
from model.model import ProductModel, TextVectorizer, BasicKerasModel
from model.preprocessor import Preprocessor
from model.utils import make_prediction
from model.utils import ClassEncoder
import time
import os

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
text_vectorizer = TextVectorizer() # vocab?
class_encoder = ClassEncoder(class_ids_path = os.path.normpath("data//products//classes.csv"))
model = BasicKerasModel("my_model.h5", threshold = 0.2)
preprocessor = Preprocessor()

@app.route('/match_products', methods=['POST'])
def predict_product():
    print(f'{time.time()}: start matching products')
    predictions = make_prediction(preprocessor, model, class_encoder, request.json)
    return [{"id" : product.get("id"), "reference_id": predictions[i]} for i,product in enumerate(request.json)]

@app.route('/api/train_new', methods=['GET'])
def train_new_mode():
    pass

if __name__ == '__main__':
    app.run(host='localhost', port='8100', debug=True)