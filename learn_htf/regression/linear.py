""" Scripts to perform linear regression methods
Based on Hastie/Tibshirani/Friedman "Elements of Statistical Learning"
Ch 3 - Linear Methods for Regression

Author: John Lee
Oct 2024
"""
# %%

import numpy as np

from learn_htf.core.matrix import Matrix
from learn_htf.core.model import Model
from learn_htf.utils.metrics import lp_norm
from learn_htf.utils.preprocessing import z_scale

# TODO handle sparse data

# %%
# follow an sklearn approach of .fit and .predict
# model.fit()


class LinearRegression(Model):
    """
    Linear regression model using least squares
    """

    def __init__(self, ridge_lambda: float = 0):
        super().__init__(ridge_lambda=ridge_lambda)

    def _fit(self, x: Matrix, y: Matrix):
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

        xfeats = x.features
        yfeats = y.features
        assert x.shape[0] == y.shape[0]

        n, p = x.shape
        x = Matrix(np.hstack([Matrix(np.ones(n)), x]))
        coef = np.linalg.solve(
            x.T @ x - self.model_params['ridge_lambda'] * Matrix(np.eye(p+1)), x.T @ y)

        samps = ['intercept'] + list(xfeats.index)
        # TODO use feature space instead of betas - need to include the intercept as well
        coef = Matrix(coef, samples=samps, features=yfeats)

        return coef

    def _predict(self, x: Matrix):
        """_summary_

        Args:
            x (Matrix): _description_

        Returns:
            _type_: _description_
        """
        xsamps = x.samples
        xfeats = x.features
        n = x.shape[0]

        x = Matrix(np.hstack([Matrix(np.ones(n)), x]))

        preds = Matrix((self.coefficients.T @ x.T).T)
        preds.samples = xsamps

        return preds

    def score(self,  y: Matrix, predictions: Matrix):
        """ TODO
        """

        return lp_norm(y, predictions, p=2)


class PrincipleComponentsRegression(LinearRegression):

    def __init__(self, n_components: int = None, ridge_lambda: float = 0.0):
        super().__init__(ridge_lambda=ridge_lambda)
        self.model_params['n_components'] = n_components

        self.x_mean = None
        self.x_std = None
        self.y_mean = None

        self.principle_components = None
        self.loadings = None
        self.singular_values = None

    def _fit(self, x, y):
        y_ = y.copy()
        x_feats = x.features

        x_, x_mean, x_std = z_scale(x.copy())
        y_mean = np.mean(y, axis=0, keepdims=True)

        y_ = y_ - y_mean

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean

        u, d, vt = np.linalg.svd(x_, full_matrices=False)

        self.loadings = vt.T
        self.principle_components = u
        self.singular_values = Matrix(d)
        n_components = x.shape[1] - 1
        n_components = min(n_components, self.model_params['n_components'])
        self.model_params['n_components'] = n_components

        self._log.info('Using n_components = %i',  n_components)

        z = u[:, :n_components] @ np.diag(d[:n_components])
        coefs = super()._fit(z, y_)

        samps = ['intercept']
        for i in range(n_components):
            samps.append(f'PC_{i}')
        coefs = Matrix(coefs.X, samples=samps)
        return coefs

    def _predict(self, x):
        # assumes that x has not been scaled
        x_ = (x - self.x_mean) / self.x_std
        n_components = self.model_params['n_components']
        z = x_ @ self.loadings[:, :n_components]

        preds = super()._predict(z)
        preds += self.y_mean
        return preds

    def score(self, x, y):
        pass


class LassoRegression(Model):
    """TODO

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__()

    def _fit(self):
        pass

    def _predict(self):
        pass

    def _score(self):
        pass


class PartialLeastSquaresRegression(Model):
    """TODO

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__()

    def _fit(self):
        pass

    def _predict(self):
        pass

    def score(self):
        pass


# %%
if __name__ == '__main__':
    from learn_htf.utils.preprocessing import expand_basis
    # TODO move tests to test directory

    xx = np.array([
        [2.5, .5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
        [3.1, .7, 2.9, 2.2, 3.9, 2.7, 1.6, 1.1, 1.6, .9],
        [1.2, .3, 1.1, .9, 1.8, 1.2, 1., .5, .7, .4]
    ]).T
    yy = np.array([10.1, 4.8, 8.9, 7.2, 12.3, 9.6, 6.5, 3.4, 5.6, 4.2])

    xx = Matrix(xx)
    yy = Matrix(yy)

    lr = LinearRegression(ridge_lambda=0)
    coefficients = lr.fit(xx, yy)
    predictions = lr.predict(xx)

    pcr = PrincipleComponentsRegression(n_components=4, ridge_lambda=0)
    coefficients = pcr.fit(xx, yy)
    predictions = pcr.predict(xx)

    # basis expansion
    xx = expand_basis(xx, [np.square, np.sqrt], include_input=True)
    lr = LinearRegression()
    lr.fit(xx, yy)
    lr.predict(xx)
