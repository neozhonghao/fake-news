from flask import Flask, request,  render_template
from waitress import serve
from src.plot import generate_plot
from src.inference import get_prediction, initialize, LemmaTokenizer
import numpy as np


app = Flask(__name__)
initializer = initialize()

@app.route('/', methods=['GET','POST'])
def index():
    """
    Renders the app homepage.
    PARAMS: None
    RETURNS: 
        render_template: IF POST - Rendering of homepage with results
                         ELSE GET - Rendering of homepage without results
                         orig_texts - Str of raw text from post
                         result - Str one of either {'Unsure', 'Fake News', 'Real News'}
                         prob - str probability of prediction being real (1 decimal)
    """
    if request.method == 'POST':
        orig_texts, real, fake, undecided, histo = predict()
        if type(orig_texts) != str or orig_texts == '':
            orig_texts = 'Regularly and thoroughly clean your hands with an alcohol-based hand rub or wash them with soap and water. Why?\
                        Washing your hands with soap and water or using alcohol-based hand rub kills viruses that may be on your hands.'
        if type(real) != np.float32 and type(real) != np.float64:
            real = np.float32(0.5)
            np.random.seed(0)
            histo = np.random.randn(1000)
        if type(histo) != np.ndarray:
            np.random.seed(0)
            histo = np.random.randn(1000)

        generate_plot(histo)
        result = 'Unsure'
        prob = real
        if undecided:
            result = 'Unsure'
            prob = ''
        else:
            if real>0.5:
                result = 'Real News'
            else:
                result = 'Fake News'
                prob = 1 - prob
            prob = '{:.1f}%'.format(prob*100)
        return render_template('index_post.html',orig_texts=orig_texts, result_text=result, prob=prob)
    return render_template('index_get.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Calls inference for prediction
    PARAMS: None (The POST object is a string variable text)
    RETURNS:
        text: Str of raw text from post.
        real: np.float32 probability of prediction being real
        fake: np.float32 probability of fake (1-real)
        undecided: np.int32 0=NOT unsure, 1=unsure
        histo: np.ndarray histogram of probabilities
    """    
    text = request.form['text']
    num_samples = request.form['num_samples']
    if text=="":
        text = 'Regularly and thoroughly clean your hands with an alcohol-based hand rub or wash them with soap and water. Why?\
                Washing your hands with soap and water or using alcohol-based hand rub kills viruses that may be on your hands.'
    if num_samples=="":
        num_samples = 100
    num_samples = int(num_samples)
    orig_texts, real, fake, undecided, histo  = get_prediction(text, num_samples=num_samples, pipeline_load=initializer)
    return text, real, fake, undecided, histo


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
