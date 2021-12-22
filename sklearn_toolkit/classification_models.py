from pathlib import Path
from typing import List

from sklearn_toolkit.interface import ClassificationModel, ModelType
from utils.data_helper import InputTextExample
from utils.log_helper import get_logger


class RandomForestModel(ClassificationModel):
    def __init__(self, model_type: ModelType = ModelType.random_forest.value,
                 model_dir: Path = None,
                 logger_name: str= 'rf_model',
                 verbose: bool = False):
        super().__init__(model_type=model_type, model_dir= model_dir, logger_name=logger_name, verbose=False)
        self.model_type = model_type
        self.model_dir = model_dir
        self.logger = get_logger(logger_name, verbose=verbose)

    def preprocessing(self):
        """preprocessing the data"""
        pass

    def load(self):
        """load model"""
        pass

    def fit(self, examples: List[InputTextExample]):
        """train the model"""
        pass

    def predict(self, examples):
        """predict test results"""
        pass

    def evaluate(self, examples: List[InputTextExample]):
        """evaluate the model"""
        pass

    def save(self):
        """save the model"""
        pass