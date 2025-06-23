# 🛡️ Differentially Private Medical Charges Classifier

&#x20;&#x20;

This repository contains a PyTorch-based training pipeline that uses **differential privacy (DP)** via Opacus to train a model on a medical dataset, classifying whether an individual's medical charges are high or not. It includes options to continue training from a saved checkpoint, visualize performance metrics, compute the confusion matrix for evaluation, and supports custom datasets.

---

## 📂 Project Structure

```
📁 dp-medical-dp-model/
├── dp_model_checkpoint.pth  # Saved model state (generated after training)
├── sample_medical_data.csv  # Input dataset
├── train_dp_model.ipynb     # Jupyter Notebook with full training code
└── README.md                # You're here!
```

---

## 🧠 What the Model Does

- 📥 **Input**: Patient info like Age, Sex, BMI, Children, Smoker status, Region.
- 📤 **Output**: Classifies whether medical charges are **above ₹15,000 (label=1)** or not (label=0).
- ⚙️ Fully connected neural network with 3 layers.
- 🔐 Training is **differentially private** using [Opacus](https://opacus.ai/).

---

## 🧰 Model Diagram

```text
[Input Layer] → [FC 128 ReLU] → [FC 64 ReLU] → [FC 2 (Logits)] → [CrossEntropy Loss]
```

---

## 🛡️ What is Differential Privacy?

Differential Privacy ensures that the model **does not memorize or leak individual data points**.

> 🔐 In this project, Opacus adds noise to gradients during training so that the final model is safe to release.

### Why DP Matters:

- Prevents re-identification attacks.
- Ensures regulatory compliance (HIPAA, GDPR).
- Useful when training on **sensitive medical data**.

### Key Concepts:

- **Epsilon (ε)**: Privacy budget. Lower = better privacy, but may reduce accuracy.
- **Delta (𝛿)**: Probability of privacy failure. Typically `1e-5`.
- **Max Grad Norm**: Controls gradient clipping for DP. Lower = stronger privacy.

---

## 🧪 Dataset Format

You must have a `sample_medical_data.csv` file in this format:

| age | sex  | bmi | children | smoker | region    | charges |
| --- | ---- | --- | -------- | ------ | --------- | ------- |
| 23  | male | 32  | 0        | yes    | southeast | 16884.9 |

> The code auto-encodes categorical values and creates a binary label:

```python
# Optional preprocessing for binary classification
from sklearn.preprocessing import StandardScaler

df['label'] = (df['charges'] > 15000).astype(int)
X = df.drop(columns=['charges', 'label'])
y = df['label'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

---

## 🚀 How to Run

### Step 1: Upload Your Dataset

```python
csv_path = "/content/your_medical_data.csv"
```

### Step 2: Choose Training Mode

```python
mode = "new"       # Start from scratch
mode = "continue"  # Load saved model and continue training
```

### Step 3: Run Code End-to-End

Includes: Data loading, preprocessing, training, saving, and evaluation.

---

## 📊 Output Metrics

- 📉 Training Loss
- ✅ Training Accuracy
- 🔐 Privacy Budget ε per Epoch
- 📈 Confusion Matrix (for evaluation)

> 📌 You can plug in **your own dataset** by matching the column format above.

---

## 🧠 Tips to Improve Accuracy ✅

| Objective                | What to Change                    | Why                                     |
| ------------------------ | --------------------------------- | --------------------------------------- |
| Improve Accuracy         | Increase `epochs`                 | Model learns patterns longer            |
|                          | Use smaller `learning rate`       | Finer weight updates                    |
|                          | Apply `StandardScaler`            | Feature normalization helps convergence |
| Reduce Loss              | Tune architecture, LR, batch size | Better optimization setup               |
| Reduce Epsilon (Privacy) | Increase `batch_size`             | Less noise per example                  |
|                          | Decrease `max_grad_norm`          | Stronger clipping = better privacy      |
|                          | Lower `target_epsilon`            | Tighter DP constraint                   |

---

## 🔍 Confusion Matrix Example

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("🧠 Confusion Matrix on Test Set")
plt.grid(False)
plt.show()
```

|          | Predicted 0 | Predicted 1 |
| -------- | ----------- | ----------- |
| Actual 0 | TN          | FP          |
| Actual 1 | FN          | TP          |

---

## 📌 Output Section

> Add screenshots of your **training plots** and **confusion matrix** here!

---

## 🧰 Requirements

Install dependencies:

```bash
pip install opacus pandas scikit-learn matplotlib
```

---

## 🙌 Contributions Welcome!

Feel free to:

- Fork the repo
- Plug in your own dataset
- Submit pull requests or issues

Let's build safer AI together 🤝

---

## 📜 License

MIT License.

---

> For any queries, feel free to contact or open an issue. Happy learning & coding with privacy! 🔐🚀

