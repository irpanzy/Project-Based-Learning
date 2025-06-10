from preprocessing import load_and_preprocess
from naive_bayes import NaiveBayesClassifier
from evaluation import print_evaluation

# 1. Load dan preprocess data
csv_path = "../data/Students Social Media Addiction.csv"
X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)

# 2. Inisialisasi dan training model
model = NaiveBayesClassifier()
model.fit(X_train, y_train)

# 3. Prediksi dan evaluasi
y_pred = model.predict(X_test)
print_evaluation(y_test, y_pred)


