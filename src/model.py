import numpy as np


class LinearRegressionScratch:
    """
    Univariate linear regression implemented
    from scratch using gradient descent.
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.theta_0 = 0.0
        self.theta_1 = 0.0

    def fit(self, X, y):
         """Train the linear regression model using gradient descent.
            """

         X = np.asarray(X).flatten()
         y = np.asarray(y)

         m = len(y)

         for _ in range(self.iterations):

            # Make predictions
            y_pred = self.theta_0 + self.theta_1 * X

              # Calculate errors
            errors = y_pred - y

            # Calculate gradients
            d_theta_0 = (2 / m) * np.sum(errors)
            d_theta_1 = (2 / m) * np.sum(errors * X)

            # Update parameters
            self.theta_0 -= self.learning_rate * d_theta_0
            self.theta_1 -= self.learning_rate * d_theta_1

         return self

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        X = np.asarray(X).flatten()

        return self.theta_0 + self.theta_1 * X
        