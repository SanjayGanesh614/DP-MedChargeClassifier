# 🛡️ Differentially Private Income Classifier (Adult Dataset)

This repository contains a PyTorch-based training pipeline that uses **differential privacy (DP)** via Opacus to train a neural network classifier on the [Adult Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income). It includes training, checkpointing, evaluation with confusion matrix, and performance visualizations. The goal is to classify whether an individual's income is **greater than \$50K**.

---

## 📂 Project Structure

```
📁 dp-income-dp-model/
├── dp_model_checkpoint.pth  # Saved model state (generated after training)
├── adult.csv                # Input dataset from Kaggle
├── train_dp_model.ipynb     # Full training code with DP
└── README.md                # You're here!
```

---

## 🧠 What the Model Does

- 📥 **Input**: Features like Age, Workclass, Education, Sex, Capital-gain, etc.
- 📤 **Output**: Predicts if income is **>50K (label=1)** or **<=50K (label=0)**.
- ⚙️ Neural network with 3 hidden layers.
- 🔐 Uses [Opacus](https://opacus.ai/) to enable training with differential privacy.

---

## 🧰 Model Diagram

```text
[Input Layer] → [FC 128 ReLU] → [FC 64 ReLU] → [FC 2 (Logits)] → [CrossEntropy Loss]
```

---

## 🛡️ What is Differential Privacy?

Differential Privacy (DP) is a technique that adds noise during training to ensure models do **not memorize or leak individual user data**.

### Why DP?

- 🧬 Ensures safety of **sensitive data**.
- ✅ Helps comply with privacy regulations (GDPR, HIPAA).
- 🤖 Enables safe model sharing.

### Key DP Concepts

- **Epsilon (ε)**: Privacy budget — smaller means stronger privacy. Typically, ε < 5 is a good goal.
- **Delta (𝛿)**: Failure probability (typically set to 1e-5).
- **Max Grad Norm**: Used to clip gradients before noise is added. Lowering this value increases privacy.
- **Privacy Engine**: Automatically clips gradients and injects noise to ensure DP constraints are met. It tracks the evolving ε during training.

---

## 🔑 Key Concepts

### Neural Network Architecture

- **Type**: Fully Connected Feedforward Neural Network (FCNN)
- **Layers**:
  - `fc1`: Linear layer with 128 neurons and ReLU
  - `fc2`: Linear layer with 64 neurons and ReLU
  - `fc3`: Output layer with 2 units for binary classification (income ≤50K or >50K)
- **Activation**: ReLU (for non-linearity)

### Learning Rate

- **Value**: `5e-4` (0.0005)
- A small learning rate ensures **stable** and **controlled updates**.
- Decreasing learning rate slows down learning but helps the model settle into a better local minimum.

### Epochs

- **Value**: `25` by default
- An epoch is one full pass over the training dataset.
- More epochs = better training (to a point) but may lead to overfitting or higher privacy cost (ε).

### Privacy Engine

- Comes from **Opacus**.
- Automatically:
  - Clips gradients to limit sensitivity
  - Adds Gaussian noise
  - Tracks ε (privacy cost) across epochs
- Works transparently with PyTorch models

---

## 🧪 Dataset Format

Use the [Adult Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income) from Kaggle and ensure it's named `adult.csv`.

| age | workclass | education | marital-status | occupation | race | sex | capital-gain | capital-loss | hours-per-week | native-country | income |
| --- | --------- | --------- | -------------- | ---------- | ---- | --- | ------------ | ------------ | -------------- | -------------- | ------ |

### Preprocessing Steps

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("adult.csv")
df.drop(columns=['fnlwgt'], inplace=True)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('income', axis=1).values.astype(np.float32)
y = df['income'].values.astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)
```

---

## 🚀 How to Run

### Step 1: Upload Your Dataset

```python
csv_path = "/content/adult.csv"
```

### Step 2: Choose Mode

```python
mode = "new"       # Start training from scratch
mode = "continue"  # Resume from last checkpoint
```

### Step 3: Run Full Code

This will do:

- Data preprocessing
- Neural network training
- Differential privacy enforcement
- Evaluation with confusion matrix

---

## 📊 Output Metrics

- 📉 Loss per epoch
- ✅ Accuracy per epoch
- 🔐 Privacy Budget ε per epoch
- 📈 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
```

---

## 📌 Tips to Improve Results

| Objective          | What to Tune                   | Why                                            |
| ------------------ | ------------------------------ | ---------------------------------------------- |
| Improve Accuracy   | Increase `epochs`              | More learning cycles                           |
|                    | Lower `learning rate`          | More stable updates and smaller learning steps |
|                    | Apply `StandardScaler`         | Normalizes feature scale                       |
| Reduce Loss        | Tune architecture & batch size | Better optimization stability                  |
| Reduce Epsilon (ε) | Increase `batch_size`          | Less noise per sample                          |
|                    | Lower `max_grad_norm`          | Stronger clipping, stronger privacy            |

---

## 🧠 Confusion Matrix Output

|             | Predicted ≤50K | Predicted >50K |
| ----------- | -------------- | -------------- |
| Actual ≤50K | TN             | FP             |
| Actual >50K | FN             | TP             |

---

## 📌 Output Section

👉 You can optionally include screenshots here of your training plots, confusion matrix, or ε chart.

---

## 🧰 Requirements

```bash
pip install opacus pandas scikit-learn matplotlib
```

---

## 🙌 Contributions Welcome!

- Fork the repo
- Use your own Kaggle datasets
- Submit issues or improvements

Let’s build private AI responsibly 💡🔐

My goal in this was to preserve data and privacy at the same time and was ok with accuracy level. But i encourage y'all to experiment i have kept some of my original code i had during my testing face with another dataset you can also use that.



i will soon update the model to read MINST datasets so stay tuned.



---

## 📜 License

MIT License

---

> Feel free to reach out for improvements, fixes, or suggestions!

