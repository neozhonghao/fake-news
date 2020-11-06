import re
import numpy as np
import pandas as pd

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class Datapipeline():
    def __init__(self): 
        pass

    ########################
    ### Helper functions ###
    ########################
    def remove_special_characters(self, text, remove_digits=True):
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern, " ", text)
        return text


    def simple_lemmatizer(self, text):
        ps = WordNetLemmatizer()
        text = ' '.join([ps.lemmatize(word.lower(), pos="v")
                        for word in text.split()])
        return text


    def remove_invalid_content(self, text):
        invalid_content = ["we 've detected that javascript is disabled in your browser. would you like to proceed to legacy twitter.",
                        "do you want to join facebook ?.",
                        "you must log in to continue ..",
                        "join this group to post and comment ..",
                        "this website is using a security service to protect itself from online attacks ..",
                        "see more.+on facebook"]
        for phrase in invalid_content:
            phrase = re.compile(phrase)
            phrase_match = bool(re.match(phrase, str(text)))
            if phrase != phrase: # test for NaN
                output = ""
                break
            elif phrase_match == True:
                output = ""
                break
            else:
                output = text
        return output


    def pipelinize(self, function, active=True):
        def list_comprehend_a_function(list_or_series, active=True):
            if active:
                return [function(i) for i in list_or_series]
            else:  # if it's not active, just pass it right back
                return list_or_series
        return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active': active})


    ##########################################################
    ### Main pipeline functions that call helper functions ###
    ##########################################################
    def preproc_train(self, path_to_real_dataset, path_to_fake_dataset):
        """
        Performs preprocessing on data, and splits into train and test sets.

        Inputs: Absolute paths to csvs containing real and fake data.
                Each csv should minimally contain the columns "title"
                and "content".
        Returns: Four numpy dense arrays: X_train, y_train, X_test, y_test
        """
        # Read real and fake datasets; concatenate them
        df_fake_news_orig = pd.read_csv(path_to_fake_dataset)
        df_real_news_orig = pd.read_csv(path_to_real_dataset)

        # Remove invalid content
        remove_inval = Pipeline(
            [('remove_invalid_content', self.pipelinize(self.remove_invalid_content))])
        df_real_news_orig["content"] = remove_inval.fit_transform(df_real_news_orig["content"])
        df_fake_news_orig["content"] = remove_inval.fit_transform(df_fake_news_orig["content"])

        # Concatenate title and content into a single string
        df_real_news = pd.DataFrame(df_real_news_orig["title"] + " " + df_real_news_orig["content"],
                                    columns=["text"])
        df_real_news["label"] = 1
        df_fake_news = pd.DataFrame(df_fake_news_orig["title"] + " " + df_fake_news_orig["content"],
                                    columns=["text"])
        df_fake_news["label"] = 0
        df_news = pd.concat([df_real_news, df_fake_news], axis=0).dropna()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(df_news['text'],
                                                            df_news['label'],
                                                            test_size=0.2,
                                                            random_state=42)

        # Main preprocessing steps
        estimators = [('remove_special_chars', self.pipelinize(self.remove_special_characters)),
                ('simple_lemmatizer', self.pipelinize(self.simple_lemmatizer)),
                ('tfid_vectorizer', TfidfVectorizer(max_features=100000,
                                                    ngram_range=(1, 1),
                                                    stop_words="english"))]
        self.pipeline = Pipeline(estimators)
        X_train = self.pipeline.fit_transform(X_train)
        X_test = self.pipeline.transform(X_test)

        # Perform under- and over-sampling on the train dataset
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        smt = SMOTE(random_state=42)
        X_train, y_train = smt.fit_resample(X_train, y_train)

        return X_train, y_train, X_test, y_test

    
    def preproc_infer(self, path_to_pred_data):
        """
        Performs preprocessing on data for inference.

        Inputs: A csv file containing the column "content"
        Returns: A numpy dense array X_pred
        """
        X_pred = pd.read_csv(path_to_pred_data)["content"]

        # Remove invalid content
        remove_inval = Pipeline(
            [('remove_invalid_content', self.pipelinize(self.remove_invalid_content))])
        X_pred = (remove_inval.fit_transform(X_pred))
        X_pred = np.array(X_pred)

        # Apply main preprocessing steps
        X_pred = self.pipeline.transform(X_pred)

        return X_pred