import os
from typing import Optional, Tuple, Union
import sys
import time
import logging

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
import sklearn.metrics as metrics
from common.trainer import *


from hpsklearn import HyperoptEstimator, svc
from hyperopt import tpe

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from utils import plot_utils


client = mlflow.client.MlflowClient()
now = round(time.time())
logger = logging.getLogger(__name__)
np.random.seed(500)

        
class SvmClassifier(Trainer):
    """
    Trainer class 
    """

    def __init__(self, experiment_name: Optional[str],
                 save_signature: Optional[bool],
                 train_path: str,
                 registered_model_version_stage : Optional[str],
                 output_path: Optional[Union[str, None]],
                 **kwargs):
        
        self.train_path = train_path

        super().__init__(experiment_name=experiment_name,
                         save_signature=save_signature,
                         registered_model_version_stage=registered_model_version_stage,
                         output_path=output_path,
                         **kwargs)
        
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # noqa
        """
        Load data and split it into training and testing sets
        """
        corpus = self.preprocess_data(self.train_path)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus['text_final'],corpus['label'],test_size=0.3)

        logger.debug(f">> X_test.type: {type(X_test)}")
        logger.debug(f">> X_test.dtypes: {X_test.dtypes}")

        return X_train, X_test, y_train, y_test
         
    def preprocess_data(self, corpus_path: str) -> pd.DataFrame:
        """
        Preprocess text
        """
        # Add the Data using pandas
        corpus = pd.read_csv(corpus_path,encoding='latin-1')
        corpus['text'].dropna(inplace=True)
        corpus['text'] = [entry.lower() for entry in corpus['text']]
        # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
        corpus['text']= [word_tokenize(entry) for entry in corpus['text']]
        # Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        for index,entry in enumerate(corpus['text']):
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words.append(word_final)
            # The final processed set of words for each iteration will be stored in 'text_final'
            corpus.loc[index,'text_final'] = str(Final_words)
        return corpus
    

    def encode_text(self, Train_Y: np.ndarray, Test_Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
        """
        # Create a label encoder
        Encoder = LabelEncoder()
        Train_Y = Encoder.fit_transform(Train_Y)
        Test_Y = Encoder.fit_transform(Test_Y)

        # Create artifacts path if it doesn't exist
        artifacts_path = os.path.join('.', 'artifacts', type(self).__name__)
        if not os.path.exists(artifacts_path):
            os.makedirs(artifacts_path)
        
        # Dump the encoders to a file
        np.savetxt(os.path.join(artifacts_path, 'Encoder_Train_Y'), Train_Y, delimiter=",")
        np.savetxt(os.path.join(artifacts_path, 'Encoder_Test_Y'), Test_Y, delimiter=",")

        return Train_Y, Test_Y
    

    def vectorize_words(self, corpus: pd.DataFrame, Train_X: np.ndarray, Test_X: np.ndarray, max_features=5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
        """
        Tfidf_vect = TfidfVectorizer(max_features=max_features)
        Tfidf_vect.fit(corpus['text_final'])

        Train_X_Tfidf = Tfidf_vect.transform(Train_X)
        Test_X_Tfidf = Tfidf_vect.transform(Test_X)

        return Train_X, Test_X
        
    @artifact_logger
    @train_wrapper
    def train(self,
              registered_model_name: Optional[str],
              plot_file: Optional[str]):
        """
        Train a model and log in mlflow
        """
        classifiers = [naive_bayes.MultinomialNB(), svc("mySVC")]
        for classifier in classifiers:
            # Find the best parameters
            model = HyperoptEstimator(classifier=classifier,
                                    algo=tpe.suggest,
                                    max_evals=20,
                                    trial_timeout=300)
            # Create model

            # Fit and predict
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)

            # MLflow params
            logger.debug("Parameters:")
            logger.debug(f"  best model: {model.best_model()}")
            logger.debug(f"  model score: {model.score( self.X_test, self.y_test )}")

            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("max_leaf_nodes", max_leaf_nodes)

            # MLflow metrics
            prec = metrics.precision(self.y_test, predictions)
            recall = metrics.recall(self.y_test, predictions)
            f1 = metrics.f1_score(self.y_test, predictions)
            roc_auc = metrics.roc_auc_score(self.y_test, predictions)
            fpr, tpr, threshold = metrics.roc_curve(y_test,  y_pred_proba)

            
            logger.debug("Metrics:")
            logger.debug(f"  prec: {prec}")
            logger.debug(f"  recall: {recall}")
            logger.debug(f"  f1: {f1}")
            logger.debug(f"  roc_auc: {roc_auc}")

            mlflow.log_metric("prec", prec)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            if plot_file:
                base_dir = os.path.dirname(plot_file)
                plot_file_temp = os.path.join(os.path.join(base_dir, type(classifier).__name__), os.path.basename(plot_file))
                plot_utils.create_plot_file(y_test_set=self.y_test, y_predicted=predictions, plot_file=plot_file_temp)
                plot_utils.plot_roc_auc(y_test_set=self.y_test, y_predicted=predictions, fpr=fpr, tpr=tpr, threshold=threshold, roc_auc=roc_auc, plot_file=plot_file_temp)
        
        return model, predictions, registered_model_name