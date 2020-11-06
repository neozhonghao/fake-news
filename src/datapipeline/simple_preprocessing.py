import numpy as np
import pandas as pd

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
import re

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer,SnowballStemmer

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# nltk.download("wordnet")
# nltk.download("stopwords")

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern, " ", text)
    return text

#Stemming the text
def simple_lemmatizer(text):
    ps = WordNetLemmatizer()
    text= ' '.join([ps.lemmatize(word.lower(), pos="v") for word in text.split()])
    return text

#set stopwords to english
stop=set(stopwords.words('english'))
# print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=True):
    #Tokenization of text
    tokenizer = ToktokTokenizer()
    #Setting English stopwords
    stopword_list = nltk.corpus.stopwords.words('english')

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]

    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def pipelinize(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            return [function(i) for i in list_or_series]
        else: # if it's not active, just pass it right back
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})



df_real_news_orig = pd.read_csv("drive/My Drive/real_news.csv")
df_fake_news_orig = pd.read_csv("drive/My Drive/fake_news.csv")

df_real_news = pd.concat([df_real_news_orig[['title']], pd.DataFrame({"label": np.ones((len(df_real_news_orig)), dtype=int)})], axis=1).reset_index(drop=True)[:len(df_fake_news_orig)]
df_fake_news = pd.concat([df_fake_news_orig[['title']], pd.DataFrame({"label": np.zeros((len(df_fake_news_orig)), dtype=int)})], axis=1).reset_index(drop=True)
df_news = pd.concat([df_real_news, df_fake_news], axis=0).reset_index(drop=True)
df_news.to_csv("drive/My Drive/covid_news.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(df['title'], df['label'], test_size=0.2, random_state=42)

estimators = [('remove_special_chars', pipelinize(remove_special_characters)),
             ('simple_lemmatizer', pipelinize(simple_lemmatizer)),
             ('remove_stopwords', pipelinize(remove_stopwords)),
             ('count_vectorizer', CountVectorizer(ngram_range=(1,1)))]
pipeline = Pipeline(estimators)

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

print('BOW_cv_train:', X_train.shape)
print('BOW_cv_test:', X_test.shape)