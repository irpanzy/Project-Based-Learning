import numpy as np


class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-6
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian_prob(self, x, mean, var):
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(self, X):
        epsilon = 1e-9
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                probs = self._gaussian_prob(x, self.means[c], self.vars[c])
                probs = np.clip(probs, epsilon, None)
                conditional = np.sum(np.log(probs))
                posteriors.append(prior + conditional)
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)
