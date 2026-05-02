#  AdPredict — Turning Social Signals into Real Results

![Cover Image](./assets/cover(3).png)

A complete end-to-end ML application predicting whether a social network user will purchase an advertised product, built using the exact code from `random_forest_classification.ipynb` on the `Social_Network_Ads.csv` dataset.

---

<!-- https://social-ads-rf.onrender.com -->

## 📁 Project Structure

```
social_ads_rf/
├── backend/
│   └── app.py                         # Flask REST API
├── frontend/
│   └── index.html                     # Web UI
├── model/
│   ├── classifier.pkl                 # Trained Random Forest model
│   ├── scaler.pkl                     # StandardScaler
│   └── stats.json                    # Accuracy, confusion matrix, etc.
├── data/
│   └── Social_Network_Ads.csv         # Original dataset (400 rows)
├── notebooks/
│   └── random_forest_classification.ipynb  # Original notebook
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend
```bash
cd backend
python app.py
# → Running on http://localhost:5000
```

### 3. Open the Frontend
Open `frontend/index.html` in your browser (just double-click).

---

## 📊 Dataset

**File:** `Social_Network_Ads.csv`  
**Rows:** 400  
**Features:** Age, EstimatedSalary  
**Target:** Purchased (0 = No, 1 = Yes)

---

## 🤖 Exact Code Used (from Notebook)

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
```

---

## 📈 Model Results

| Metric | Score |
|---|---|
| **Accuracy** | **91.0%** |
| Precision | 84.85% |
| Recall | 87.50% |
| F1 Score | 86.15% |

**Confusion Matrix (100 test samples):**
```
[[63  5]
 [ 4 28]]
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Model stats & confusion matrix |
| POST | `/api/predict` | Single prediction |
| POST | `/api/batch_predict` | Batch predictions |

### Example Predict Request
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 42, "salary": 90000}'
```

### Response
```json
{
  "success": true,
  "prediction": 1,
  "label": "Purchased",
  "probability_not_purchased": 0.1,
  "probability_purchased": 0.9,
  "confidence": 0.9
}
```
