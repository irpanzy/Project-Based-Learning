import numpy as np


class NaiveBayesClassifier:
    def fit(self, X, y):
        """
        Melatih model Naive Bayes dengan data training
        X: fitur training data
        y: label training data
        """
        # Mendapatkan semua kelas unik dalam data
        self.classes = np.unique(y)

        # Inisialisasi dictionary untuk menyimpan parameter
        self.means = {}  # rata-rata setiap fitur per kelas
        self.vars = {}  # variansi setiap fitur per kelas
        self.priors = {}  # probabilitas prior setiap kelas

        # Hitung parameter untuk setiap kelas
        for c in self.classes:
            # Ambil data yang termasuk kelas c
            X_c = X[y == c]

            # Hitung rata-rata setiap fitur untuk kelas c
            self.means[c] = np.mean(X_c, axis=0)

            # Hitung variansi + smoothing untuk menghindari divide by zero
            self.vars[c] = np.var(X_c, axis=0) + 1e-6

            # Hitung probabilitas prior kelas c
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
