#Standard library imports
import os
from typing import Optional, Tuple, Union
import logging

# Third-party imports
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import torch

# Local imports
from common.trainer import artifact_logger_factory, train_wrapper
from common.trainer import Trainer as BaseTrainer
from utils import plot_utils


client = mlflow.client.MlflowClient()
logger = logging.getLogger(__name__)
np.random.RandomState(500)

# https://huggingface.co/docs/transformers/tasks/sequence_classification


class BertClassifier(BaseTrainer):
    """
    Trainer class 
    """

    def __init__(self, experiment_name: Optional[str],
                 save_signature: Optional[bool],
                 data_path: str,
                 registered_model_version_stage : Optional[str],
                 output_path: Optional[Union[str, None]],
                 **kwargs):
        
        self.data_path = data_path

        super().__init__(experiment_name=experiment_name,
                         save_signature=save_signature,
                         registered_model_version_stage=registered_model_version_stage,  # noqa
                         output_path=output_path,
                         model_name_run= self.__class__.__name__,
                         **kwargs)
        self.tokenize_data()
         
    def preprocess_function(self, examples):
        """
        Preprocess function for tokenizer
        """
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    def infer_signature(self):
        """
        Create signature for inference
        """
        predictions = self.get_predictions()
        return infer_signature(self.X_train, predictions) if self.save_signature else None # noqa
    
    def get_predictions(self):
        """
        Get predictions from the model
        """
        if self.model:
            # Load model and evaluate predictions
            model = self.model
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                predictions = model(self.tokenizer_data["train"])
        else:
            logger.error("Model is not loaded")

        return predictions
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # noqa
        """
        Load data and split it into training and testing sets
        """
        # This is required forthe preprocess_function function
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        data = pd.read_csv(self.data_path, encoding="latin")

        X_train, X_test, y_train, y_test = model_selection.train_test_split(data['text'],data['label'],test_size=0.3) # noqa
        df_train = pd.DataFrame({'text': X_train, 'label': y_train})
        df_test = pd.DataFrame({'text': X_train, 'label': y_train})
        self.data = DatasetDict(
            {"train": Dataset.from_pandas(df_train),
             'test': Dataset.from_pandas(df_test),})
           
    def tokenize_data(self) -> None:
        """
        Tokenize data input
        """
        data = self.data
        self.tokenizer_data = data.map(self.preprocess_function, batched=True)
        
        # Remove unnecessary columns, this is required for the trainning process
        # columns = [col_name if col_name not in ['label', 'text'] else '' for col_name in self.tokenizer_data["train"].column_names]        
        # columns = list(filter(None, columns))
        # columns = ['text', 'label']

        # Create tokenizer
        self.tokenizer_data = self.tokenizer_data.remove_columns(self.data["train"].column_names)
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.tokenizer = tokenizer

        # Encode labels
        le = LabelEncoder()
        names = data['train']['label']
        names.extend(x for x in data['test']['label'] if x not in names)
        le.fit(names)

        # Generate dictionries of {ID : LABEL} and {LABEL : ID}
        self.label2id = dict(zip(le.classes_, [int(x) for x in le.transform(le.classes_)]))
        self.id2label = dict(zip([int(x) for x in le.transform(le.classes_)], le.classes_))
        self.num_labels = len(self.label2id.keys())

        self.data = data

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for Trainer class
        """
        accuracy = evaluate.load('accuracy')
        predictions, labels = eval_pred

        predictions = np.argmax(predictions, axis=1)

        return accuracy.compute(predictions=predictions, references=labels)

    @artifact_logger_factory("BertClassifier")
    @train_wrapper
    def train(self,
              registered_model_name: Optional[str],
              plot_file: Optional[str]):
        """
        Train a model and log in mlflow
        """
        # Create model
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
            )
        # https://wandb.ai/biased-ai/huggingface/reports/Training-Transformers-for-Text-Classification-on-HuggingFace--Vmlldzo3OTc3MDg
        training_args = TrainingArguments(
            output_dir='huggingface_output',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            # label_names=list(self.label2id.keys())
            )
        
        # Fit and predict
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenizer_data['train'],
            eval_dataset=self.tokenizer_data['test'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        # https://github.com/huggingface/transformers/issues/9636
        trainer.train()        
        
        logits = []
        with torch.no_grad():
            for x in self.y_test:
                inputs = self.tokenizer(x, return_tensors="pt")
                logits.append(model(**inputs).logits)
        
        predictions = [logit.argmax().item() for logit in logits]

        # MLflow metrics
        prec = metrics.precision_score(self.y_test, predictions)
        recall = metrics.recall_score(self.y_test, predictions)
        f1 = metrics.f1_score(self.y_test, predictions)
        roc_auc = metrics.roc_auc_score(self.y_test, predictions)
        
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
            name = self.__class__.__name__
            base_dir = os.path.dirname(plot_file)
            plot_file_temp = os.path.join(os.path.join(base_dir, name), os.path.basename(plot_file))
            plot_utils.create_plot_file(y_test_set=self.y_test, y_predicted=predictions, plot_file=plot_file_temp) # noqa
        
        self.model = model

        return registered_model_name