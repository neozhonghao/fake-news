import numpy as np
import torch
import torch.nn as nn
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro

import joblib
import sys

torch.manual_seed(100)
np.random.seed(100)

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
# nltk.download('punkt')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

class BaseModel(nn.Module): # LogisticRegression
    def __init__(self, num_features):
        super(BaseModel, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

def model(x, y=None, p=None):
    # Create priors over the parameters
    loc, scale = torch.zeros(1, p), 10 * torch.ones(1, p)
    bias_loc, bias_scale = torch.zeros(1), 10 * torch.ones(1)
    w_prior = Normal(loc, scale).independent(1)
    b_prior = Normal(bias_loc, bias_scale).independent(1)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", base_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    with pyro.plate("map", N):
        x_data = x
        y_data = y
        
        model_logits = lifted_reg_model(x_data).squeeze(-1)
        pyro.sample("obs", Bernoulli(logits=model_logits, validate_args=True), obs=y_data.squeeze())

softplus = torch.nn.Softplus()

def guide(x, y=None, p=None):
    # define our variational parameters
    w_loc = torch.randn(1, p)
    # note that we initialize our scales to be pretty narrow
    w_log_sig = torch.as_tensor(-8.0 * torch.ones(1, p) + 0.05 * torch.randn(1, p))
    b_loc = torch.randn(1)
    b_log_sig = torch.as_tensor(-8.0 * torch.ones(1) + 0.05 * torch.randn(1))
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_loc)
    sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))
    # guide distributions for w and b
    w_dist = Normal(mw_param, sw_param).independent(1)
    b_dist = Normal(mb_param, sb_param).independent(1)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
    # overload the parameters in the module with random samples
    # from the guide distributions
    lifted_module = pyro.random_module("module", base_model, dists)
    # sample a regressor (which also samples w and b)
    return lifted_module()

# Training loop
def train():
    """
    training function for maximizing the ELBO
    """
    pyro.clear_param_store()
    x = torch.tensor(np.array(X_train_tfidf.todense()), dtype=torch.float)
    y = torch.tensor(np.array(y_train_tfidf).reshape(-1,1), dtype=torch.float)
    num_features = x.shape[1]
    num_iterations = 2000
    for j in range(num_iterations):
        loss = svi.step(x, y, num_features)
    if j % (num_iterations / 10) == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(N)))

def give_uncertainties(x, num_samples):
    """
    args:
        x: The numpy BOW matrix
        num_samples: The number of predictions to sample
    return:
        yhats: The sampled prediction probabilities
    """
    num_features = x.shape[1]
    sampled_models = [guide(None, None, num_features) for _ in range(num_samples)]
    yhats = [model(x).data.detach().cpu().numpy() for model in sampled_models]
    return np.array(yhats)

def run_inference_single(x, orig_texts=None, num_samples=10):
    """
    args:
        x: The numpy BOW matrix
        orig_texts: The original texts to be predicted
        num_samples: The number of predictions to sample
    return:
        orig_texts: The original texts to be predicted e.g. "Real or fake?"
        real: The probability of being a real news e.g. 0.991
        fake: The probability of being a fake news e.g. 0.881
        undecided: The network is uncertain e.g. False
        histo: The sampled probability in 1D numpy array e.g. [0.11, 0.004, ...]
    """
    y = give_uncertainties(x, num_samples)
    histo = y.reshape(-1)
    prob = np.percentile(histo, 50) # sampling median probability
    
    # Inference portion
    predicted = prob > 0.5
    real = prob
    fake = 1 - real
    undecided = np.abs(real - fake) < 0.5

#     print("real %.5f, fake %.5f, undecided %d" % (real, fake, undecided))
#     print(orig_texts)       
  
    return orig_texts, real, fake, undecided, histo

def save_checkpoint():
    pyro.get_param_store().save("misc/model_checkpoint_flat.pt")
    
def load_checkpoint(net):
    pyro.module('net', net, update_module_params=True)
    pyro.get_param_store().load("misc/model_checkpoint_flat.pt")

def load_pipeline():
    pipeline_load = joblib.load('misc/pipeline_tfidf.joblib')
    return pipeline_load

def initialize():
    """
    Initialize the pipeline transformation
    return:
        pipeline_load: load the text preprocessing pipeline
    """
    pipeline_load = load_pipeline()
    return pipeline_load
    
## Call this function from Flask app.py
def get_prediction(input_text, num_samples, pipeline_load):
    input_text = [input_text] # Expected format for the model
    input_text_transform = np.array(pipeline_load.transform(input_text).todense())
    to_predict = torch.tensor(input_text_transform, dtype=torch.float) # (1 x 3555)
    orig_texts, real, fake, undecided, histo = run_inference_single(x=to_predict, orig_texts=input_text, num_samples=num_samples)
    return orig_texts, real, fake, undecided, histo

# # Training
# num_features = 3555 #X_train_tfidf.shape[1] #100
# base_model = BaseModel(num_features)
# optim = Adam({"lr": 0.005})
# svi = SVI(model, guide, optim, loss=Trace_ELBO())

# Initialize model
num_features = 3555
print("Initializing base model")
base_model = BaseModel(num_features)
load_checkpoint(base_model)

if __name__ == '__main__':
    input_text = sys.argv[1]# ["This is real news"]
    num_samples = int(sys.argv[2]) # The number of predictions to sample e.g. 200
    print(get_prediction(input_text, num_samples, initialize()))