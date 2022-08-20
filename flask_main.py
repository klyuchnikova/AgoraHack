from flask import Flask, request
from model.model import ProductModel, TextVectorizer
from model.preprocessor import Preprocessor
import time

app = Flask(__name__)
text_vectorizer = TextVectorizer() # vocab?
model = ProductModel(text_vectorizer, threshold = 0.2)
preprocessor = Preprocessor()

def make_prediction(preprocessor, model, products):
    preprocessed_data = preprocessor.preprocess(products)
    model_predictions = model.predict(preprocessed_data)


@app.route('/match_products', methods=['POST'])
def predict_product():
    print(f'{time.time()}: start matching products')
    predictions = make_prediction(request.form)
    return [{"id" : product.get("id"), "reference_id": predictions[i]} for i,product in enumerate(request.form)]

@app.route('/api/train_new', methods=['GET'])
def train_new_mode():
    pass

if __name__ == '__main__':
    app.run(host='localhost', port='8100', debug=True)