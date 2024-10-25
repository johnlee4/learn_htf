# %%
import numpy as np
from learn_htf.core.matrix import Matrix
from learn_htf.core.model import Model


def softmax(z):
    # Subtracting max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class LinearDiscriminantAnalysis(Model):
    """TODO

    Args:
        Model (_type_): _description_
    """

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)
        self.eigenvectors = None
        self.n_groups = None
        self.groups = None
        self.cov = None

        self.means = {}
        self.priors = {}

    def _fit(self, x: Matrix, y: Matrix):
        overall_mean = np.mean(x, axis=0, keepdims=True)

        # Get unique class labels
        groups = np.unique(y.X)
        n_groups = len(groups)
        self.n_groups = n_groups
        self.groups = groups

        n_y = y.shape[1]
        n_samples, n_features = x.shape

        # TODO currently only done for the first column of y.
        ycol = 0

        self.means = np.zeros((n_groups, n_features))
        self.priors = np.zeros((n_groups, 1))
        # initialize the within class variances
        sw = np.zeros((n_features, n_features))
        # initialize the between class variances
        sb = np.zeros((n_features, n_features))

        for k in groups:
            x_k = x[y[:, ycol].X == k, :]
            n_k = x_k.shape[0]
            mu_k = x_k.mean(axis=0, keepdims=True)

            self.means[k, :] = mu_k
            self.priors[k, :] = n_k / x.shape[0]
            # tmp = np.cov(x_k, rowvar= False)
            # compute the within class variance
            sw += (x_k - mu_k).T @ (x_k - mu_k)

            mu_dif = mu_k - overall_mean
            sb += n_k * mu_dif.T @ mu_dif

        eigvals, eigvecs = np.linalg.eig(np.linalg.solve(sw, sb))
        sorted_indices = np.argsort(abs(eigvals))[::-1]
        self.eigenvectors = Matrix(eigvecs[:, sorted_indices])
        self.cov = sw / (x.shape[0] - n_groups)

        return self.cov

    def _predict(self, x: Matrix):

        if self.model_params['n_components'] is None:
            self.model_params['n_components'] = self.n_groups - 1

        n_components = int(np.minimum(
            self.model_params['n_components'], self.eigenvectors.shape[1]))
        w = self.eigenvectors[:, :n_components]

        x_proj = Matrix(x @ w)
        inv_cov = w.T @  np.linalg.inv(self.cov) @ w

        mns = self.means @ w

        term2 = np.diag((mns @ inv_cov @ mns.T).X) / 2
        term1 = x_proj @ inv_cov  @ mns.T
        log_priors = np.log(self.priors)

        discriminants = np.real(
            Matrix(term1 - term2[np.newaxis, :] + log_priors.T))
        preds = self.groups[np.argmax(
            discriminants, axis=1, keepdims=True)]

        return preds

    def score(self):
        pass


class QuadraticDiscriminantAnalysis(Model):
    """TODO

    Args:
        Model (_type_): _description_
    """

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)
        self.eigenvectors = None
        self.n_groups = None
        self.n_features = None

        self.groups = None
        self.cov = None
        self.means = None
        self.priors = None

    def _fit(self, x: Matrix, y: Matrix):

        # Get unique class labels
        groups = np.unique(y.X)
        n_groups = len(groups)

        n_y = y.shape[1]
        n_samples, n_features = x.shape

        self.groups = groups
        self.n_groups = n_groups
        self.n_features = n_features
        self.means = np.zeros((n_groups, n_features))
        self.priors = np.zeros((n_groups, 1))
        self.cov = np.zeros((n_groups, n_features, n_features))

        # TODO currently only done for first column of y
        ycol = 0

        for k in groups:
            x_k = x[y[:, ycol].X == k, :]
            n_k = x_k.shape[0]
            mu_k = x_k.mean(axis=0, keepdims=True)

            self.means[k, :] = mu_k
            self.priors[k, :] = n_k / x.shape[0]
            self.cov[k, :, :] = np.cov(x_k, rowvar=False)

    def _predict(self, x: Matrix):
        n_samples = x.shape[0]
        n_groups = self.n_groups

        discriminants = np.zeros((n_samples, n_groups))

        if self.model_params['n_components'] is None:
            self.model_params['n_components'] = self.n_features

        n_components = int(np.minimum(
            self.model_params['n_components'], x.shape[1]))

        for idx in range(self.n_groups):
            cov = self.cov[idx, :, :]
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            eig_idx = np.argsort(eigenvalues)[::-1]
            top_k_eigenvectors = eigenvectors[:, eig_idx[:n_components]]

            x_proj = x @ top_k_eigenvectors     # shape (nsamp , topk)
            mu_k = self.means[idx, :] @ top_k_eigenvectors  # shape (topk,)
            cov_k = top_k_eigenvectors.T @ cov @ top_k_eigenvectors

            logdetcov_k = np.log(np.linalg.det(cov_k))
            inv_cov = np.linalg.inv(cov_k)

            centered_x = x_proj - mu_k
            term1 = -1/2 * Matrix(np.diag(centered_x @ inv_cov @ centered_x.T))
            term2 = -1/2 * logdetcov_k
            term3 = np.log(self.priors[idx, :])

            discriminants[:, idx] = (term1 + term2 + term3).flatten()

        preds = Matrix(self.groups[np.argmax(discriminants, axis=1)])

        return preds

    def score(self):
        pass

# TODO
# implement with newton raphson - RAM heavy though


class LogisticRegression(Model):
    """TODO

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.W = None
        self.b = None

    def forward_propagation(self, x, W, b):
        Z = x @ W + b  # Linear step (XW + b)
        A = softmax(Z)  # Apply softmax to get probabilities
        return A

    def compute_cost(self, A, y):

        m = y.shape[0]
        log_probs = -np.log(A[range(m), y])
        cost = np.sum(log_probs) / m
        return cost

    def compute_gradients(self, x: Matrix, A: Matrix, y: Matrix):
        m = x.shape[0]
        y_one_hot = np.eye(A.shape[1])[y]
        y_one_hot = y_one_hot[:, 0, :]
        dz = A.X - y_one_hot
        dw = (x.T @ dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        return dw, db

    def update_parameters(self, W, b, dw, db, learning_rate=0.01):
        W -= learning_rate * dw
        b -= learning_rate * db
        return W, b

    def _fit(self, x: Matrix, y: Matrix):
        n_samples, n_features = x.shape
        groups = np.unique(y.X)
        n_groups = len(groups)
        learning_rate = 0.01
        n_iterations = 1001

        # Assign initial random weights (n_features x n_classes)
        W = np.random.normal(0, 1, size=(n_features, n_groups))
        # initialize beta vector
        b = np.zeros((1, n_groups))
        prev_cost = 1e10
        for i in range(n_iterations):
            # Forward propagation
            A = self.forward_propagation(x, W, b)
            cost = self.compute_cost(A, y)

            if cost > prev_cost:
                learning_rate = learning_rate/2

            if learning_rate < 1e-10:
                break

            prev_cost = cost

            dw, db = self.compute_gradients(x, A, y)  # Compute gradients
            W, b = self.update_parameters(
                W, b, dw, db, learning_rate)  # Update parameters

            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")

        self.W = W
        self.b = b
        return W, b

    def _predict(self, x):
        A = self.forward_propagation(x, self.W, self.b)
        preds = Matrix(np.argmax(A, axis=1))
        return preds

    def score(self):
        pass


# %%
if __name__ == '__main__':

   # Parameters
    n_samples = 1000  # Number of samples
    n_features = 10   # Number of features (dimensions)
    n_classes = 3     # Number of classes

    # Randomly generate feature data
    x = np.random.randn(n_samples, n_features)

    # Create class labels based on a simple rule:
    # For example, summing the features and using a modulo operation to assign classes
    y = (np.sum(x, axis=1) > np.median(np.sum(x, axis=1))).astype(int)
    y += (np.sum(x[:, :n_features // 2], axis=1) >
          np.median(np.sum(x[:, :n_features // 2], axis=1))).astype(int)

    x = Matrix(x)
    y = Matrix(y)

    lda = LinearDiscriminantAnalysis()
    lda.fit(x, y)
    preds = lda.predict(x)
    print(np.sum(Matrix(preds) == y, keepdims=True) / len(y))

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x, y)
    preds = qda.predict(x)
    print(np.sum(Matrix(preds) == y, keepdims=True) / len(y))

    lr = LogisticRegression()
    lr.fit(x, y)
    preds = lr.predict(x)
    print(np.sum(Matrix(preds) == y, keepdims=True) / len(y))


# %%
