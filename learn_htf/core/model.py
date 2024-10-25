"""
model.py
base model 
"""


from pathlib import Path
from typing import Union
from abc import ABC, abstractmethod

from learn_htf.utils import io, log
from learn_htf.core.matrix import Matrix


def load_model(filepath: Union[Path, str]):
    filepath = Path(filepath)
    print("pretending to load the model")
    # needs to load dict fields and populate __dict__
    # needs to read the model name and instantiate an instance of that model
    # needs to initialize the coefficents and set .fitted to True


class Model(ABC):
    """
    Model base class for scalability and reproducibility
    """

    def __init__(self, **kwargs):
        self.model_type = self.__class__.__name__
        self._log = log.get_log(self.__class__.__name__)
        log.configure_logging()

        self.coefficients = None
        self.fitted = False
        self.predictions = None

        self.model_params = kwargs

    @abstractmethod
    def _fit(self, x: Matrix, y: Matrix):
        """_summary_

        Args:
            x (Matrix): _description_
            y (Matrix): _description_
        """

    @abstractmethod
    def _predict(self, x: Matrix):
        """_summary_

        Args:
            x (Matrix): _description_
        """

    @abstractmethod
    def score(self,  y: Matrix, predictions: Matrix):
        """_summary_

        Args:
            x (Matrix): _description_
            y (Matrix): _description_
        """

    def fit(self, x: Matrix, y: Matrix):
        """_summary_

        Args:
            x (Matrix): _description_
            y (Matrix): _description_
        """

        self._log.info("Fitting %s", self)

        # returns a Matrix object
        self.coefficients = self._fit(x, y)
        self.fitted = True

        return self.coefficients

    def predict(self, x: Matrix):
        """_summary_

        Args:
            x (Matrix): _description_

        Returns:
            _type_: _description_
        """
        x = Matrix(x)
        if self.fitted:
            self._log.info("Predicting %s", self)
            self.predictions = self._predict(x)
        else:
            raise ValueError(f'{self.model_type} has not been fitted yet!')
        return self.predictions

    def save_model(self, filepath: Union[Path, str]):
        """_summary_

        Args:
            filepath (Union[Path, str]): _description_
        """
        filepath = Path(filepath)
        if self.fitted:
            print("pretending to save the model")

            # requires that a model be fitted
            # needs to save __dict__ fields
            # needs to store the model name
            # needs to store the coefficents

        else:
            raise ValueError("Model is not fitted! No need to save it")

    def get_params(self):
        """returns the model paramters

        Returns:
            dict: Model parameters 
        """
        return ', '.join(f'{k}={v}' for k, v in self.model_params.items() if not k.startswith('_'))

    def __repr__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return f"{self.model_type}({self.get_params()})"
