from learn_htf.core.model import Model
from pathlib import Path
from typing import Union


def load_model(filepath: Union[Path, str]):
    filepath = Path(filepath)
    print("pretending to load the model")
    # needs to load dict fields and populate __dict__
    # needs to read the model name and instantiate an instance of that model
    # needs to initialize the coefficents and set .fitted to True


def save_model(model: Model, filepath:  Union[Path, str]):

    filepath = Path(filepath)
    if model.fitted:
        print("pretending to save the model")

        # requires that a model be fitted
        # needs to save __dict__ fields
        # needs to store the model name
        # needs to store the coefficents

    else:
        raise ValueError("Model is not fitted! No need to save it")
