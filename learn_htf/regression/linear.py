""" Scripts to perform linear regression methods
Based on Hastie/Tibshirani/Friedman "Elements of Statistical Learning"
Ch 3 - Linear Methods for Regression

Author: John Lee
Oct 2024
"""

import numpy as np
from scipy.linalg import svd

from learn_htf.core.matrix import Matrix

# TODO Principle component regression
# TODO SVD/cholesky/QR decomp
# TODO handle sparse data

# follow an sklearn approach of .fit and .predict
# model.fit()


class LinearRegression:
    """
    Linear regression model using least squares
    """

    def __init__(self):
        self.coefficients = None
        self.score = None
        self.trained = False

    def fit(self, x: Matrix, y: Matrix, ridge_lambda: float = 0):
        """
        Calculates the coefficients (B_i) of a linear fit
            f(X) = B_0 + Sum( X_i B_i )
        that minimizes ||f(X) - y||


        Args:
            x (np.ndarray): Size N x P
            y (np.ndarray): Size N x K

        Returns:
            _type_: Coefficients of linear fit: Size M+1 x K
        """
        assert x.shape[0] == y.shape[0]

        n, p = x.shape
        x = np.hstack([np.ones((n, 1)), x])
        coef = np.linalg.inv(x.T @ x - ridge_lambda * np.eye(p+1)) @ x.T @ y

        self.coefficients = coef
        self.trained = True

    def predict(self):
        """ TODO
        """
        pass

    def score(self, yhat, y):
        """ TODO
        """
        pass


def least_squres_fit(x: Matrix, y: Matrix, ridge_lambda: float = 0):

    return coef


def least_squares_eval(coef, x):
    N, p = x.shape
    yhat = x@coef
    variance = 1/(N - p-1) * np.sum((yhat - y)**2, 1)

    return yhat


def principle_components_regression(x: np.ndarray, y: np.ndarray, pct_variance=.9, x_centered=False):

    b0 = y.mean(axis=0)

    if not x_centered:
        x = x - x.mean(axis=0)
    u, d, vt = svd(x, full_matrices=False)

    # get the first PC where we reach pct_variance
    explanined_variance = d**2 / d.sum(0)
    npc = np.nonzero(np.cumsum(explanined_variance) > pct_variance)[0][0]

    # x@vt.T
    # (y - b0 + u[:, :npc]@u[:, :npc].T@y)
