import numpy as np


class NaiveBayesClassifier:
    def fit(self, X, y):

        self.classes = np.unique(y)  # Mendapatkan kelas unik [0, 1]

        self.means = {}  # Rata-rata setiap fitur per kelas
        self.vars = {}  # Variansi setiap fitur per kelas
        self.priors = {}  # Probabilitas prior setiap kelas

        for c in self.classes:

            X_c = X[y == c]  # Data untuk kelas c

            # Hitung rata-rata (μ) setiap fitur
            self.means[c] = np.mean(X_c, axis=0)

            # Hitung variansi (σ²) dengan smoothing
            self.vars[c] = np.var(X_c, axis=0) + 1e-6

            # Hitung probabilitas prior P(kelas)
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

                # Log probabilitas prior
                prior = np.log(self.priors[c])

                # Hitung probabilitas setiap fitur
                probs = self._gaussian_prob(x, self.means[c], self.vars[c])

                # Clipping untuk menghindari log(0)
                probs = np.clip(probs, epsilon, None)

                # Log conditional probability
                conditional = np.sum(np.log(probs))

                # Posterior = prior + conditional (dalam log space)
                posteriors.append(prior + conditional)
                
            # Pilih kelas dengan posterior probability tertinggi
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)
