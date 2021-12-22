from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List

from utils.data_helper import InputTextExample, InputNumericExample
from utils.log_helper import get_logger

class ModelType(Enum):
    keyword_model = "keyword_model"
    regex_model = "regex_model"
    linear_regression = "linear regression"
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    support_vector_machine = "svm"

class ClassificationModel(ABC):
    def __init__(self, model_type: ModelType,
                 model_dir: Path,
                 logger_name: str,
                 verbose: bool = False):
        self.model_type = model_type
        self.model_dir = model_dir
        self.logger = get_logger(logger_name, verbose=verbose)

    @abstractmethod
    def preprocessing(self):
        """preprocessing the data"""
        pass

    @abstractmethod
    def load(self):
        """load model"""
        pass

    @abstractmethod
    def fit(self, examples: List[InputTextExample]):
        """train the model"""
        pass

    @abstractmethod
    def predict(self, examples):
        """predict test results"""
        pass

    @abstractmethod
    def evaluate(self, examples: List[InputTextExample]):
        """evaluate the model"""
        pass

    @abstractmethod
    def save(self):
        """save the model"""
        pass

class RegressionModel(ABC):
    def __init__(self, model_type: ModelType,
                 model_dir: Path,
                 logger_name: str,
                 verbose: bool = False):
        self.model_type = model_type
        self.model_dir = model_dir
        self.logger = get_logger(logger_name, verbose=verbose)

    @abstractmethod
    def preprocessing(self):
        """preprocessing the data"""
        pass

    @abstractmethod
    def load(self):
        """load model"""
        pass

    @abstractmethod
    def fit(self, examples: List[InputNumericExample]):
        """train the model"""
        pass

    @abstractmethod
    def predict(self, examples):
        """predict test results"""
        pass

    @abstractmethod
    def evaluate(self, examples: List[InputNumericExample]):
        """evaluate the model"""
        pass

    @abstractmethod
    def save(self):
        """save the model"""
        pass

class RuleBaseModel(ABC):
    def __init__(self, model_type: ModelType,
                 model_dir: Path,
                 logger_name: str,
                 verbose: bool = False):
        self.model_type = model_type
        self.model_dir = model_dir
        self.logger = get_logger(logger_name, verbose=verbose)

    @abstractmethod
    def load(self, rules):
        """load model"""
        pass

    @abstractmethod
    def predict(self, examples: List[InputTextExample]):
        pass