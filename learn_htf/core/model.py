"""
model.py
base model 
"""


from pathlib import Path
from typing import Union
from abc import ABC, abstractmethod

from learn_htf.utils import io, log
from learn_htf.core.matrix import Matrix



class Model(ABC):
    """
    Model base class for scalability and reproducibility
    """

    def __init__(self, **kwargs):
        self.model_type = self.__class__.__name__  
        self._log = log.get_log()

        self.coefficients = None
        self.fitted = False

        self.model_params = kwargs

    @abstractmethod
    def _fit(self, x: Matrix, y: Matrix):
        """_summary_

        Args:
            x (Matrix): _description_
            y (Matrix): _description_
        """

    @abstractmethod
    def _predict(self, y: Matrix):
        """_summary_

        Args:
            y (Matrix): _description_
        """

    @abstractmethod
    def score(self, x: Matrix, y: Matrix):
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
        self._log.info("Fitting %s", self )

        # returns a Matrix object
        self.coefficients = self._fit(x,y)
        self.fitted = True

        self._log.info("Got %s", self.coefficients )

    def predict(self, y: Matrix):
        """_summary_

        Args:
            y (Matrix): _description_

        Returns:
            _type_: _description_
        """
        self._log.info("Predicting %s", self )
        return self._predict(y)

    def save_model(self, filepath: Union[Path, str]):
        """_summary_

        Args:
            filepath (Union[Path, str]): _description_
        """
        filepath = Path(filepath)
        io.save_model(self, filepath)

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
