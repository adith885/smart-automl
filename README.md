# 🚀 Smart AutoML (Scikit-learn Based)

A lightweight AutoML framework built using only Scikit-learn that automatically:

- Detects ML task (classification / regression)
- Handles preprocessing (numeric + categorical)
- Trains multiple models
- Selects best model automatically
- Handles errors safely (no crashes)
- Works with raw CSV datasets

---

# 📌 Features

- 🔍 Auto task detection
- ⚙️ Automatic preprocessing (missing values + encoding)
- 📊 Multiple model training
- 🏆 Best model selection
- 🧠 Safe cross-validation handling
- ❌ Error-safe execution (no runtime crashes)
- 📦 Works with real-world messy datasets

---


# ⚙️ Installation

```bash
git clone https://github.com/adith885/smart-automl.git
cd smart-automl

pip install -r requirements.txt
```

---

# 🚀 Quick Start

```python
import pandas as pd
from smartautoml.automl import SmartAutoML

df = pd.read_csv("dataset.csv")

automl = SmartAutoML(cv=3)

automl.fit(df, target_column="Target")

predictions = automl.predict(df.drop("Price", axis=1))

print(predictions)
```

---

# 🧠 How It Works

### Step 1: Detect Task
- Classification
- Regression

### Step 2: Preprocessing
- Missing value handling
- OneHotEncoding
- Feature scaling

### Step 3: Model Training
Classification:
- Logistic Regression
- Random Forest
- SVM

Regression:
- Linear Regression
- Random Forest
- SVR

### Step 4: Model Selection
Best model chosen automatically

---

# ⚠️ Error Handling

```
❌ Error: Target column not found
⚠️ Using fallback CV
❌ Training failed for model svm
```

---


# 🧪 Supported Tasks

- Classification
- Regression

---

# 📦 Requirements

```
numpy
pandas
scikit-learn
joblib
```

---

# 🚀 Future Improvements

- Auto hyperparameter tuning
- Feature importance report
- Streamlit UI dashboard
- FastAPI deployment
- SHAP explainability

---

# 👨‍💻 Author

Built by **Adith**  
GitHub: https://github.com/adith885
```