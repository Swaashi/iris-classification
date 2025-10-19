# Iris Flower Classification

**Goal:** Classify iris flowers (Setosa / Versicolor / Virginica) using sepal & petal measurements.

**Dataset:** Iris dataset (Kaggle / UCI). 150 samples, 4 features.

**Models tried:** Logistic Regression, K-Nearest Neighbors (K=5), Support Vector Machine (linear).  
**Best result:** All three models achieved 100% accuracy on the test split (80/20) for this run.

**How to run (local / Colab):**
1. Clone the repo and place `svm_iris_model.pkl` and `labelencoder_iris.pkl` in the project folder.
2. Install requirements: `pip install -r requirements.txt` (includes scikit-learn, pandas).
3. Run predict script:
   ```python
   import joblib, numpy as np
   svm = joblib.load("svm_iris_model.pkl")
   le  = joblib.load("labelencoder_iris.pkl")
   sample = np.array([[5.8, 2.7, 5.1, 1.9]])
   pred = svm.predict(sample)[0]
   # if pred is numeric: pred_label = le.inverse_transform([pred])[0]
   # else pred is already the species string
