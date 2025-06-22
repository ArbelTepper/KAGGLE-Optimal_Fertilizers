# Predicting Optimal Fertilizers - Kaggle Playground Series

## Overview

This repository contains my solution for the [Predicting Optimal Fertilizers](https://www.kaggle.com/competitions/playground-series-s5e6/) Kaggle Playground competition. The goal is to recommend the best fertilizer(s) for different weather, soil, and crop conditions using a deep learning approach.

---

## Competition Description

**Objective:**  
Select the best fertilizer for different weather, soil conditions, and crops.

**Submission Format:**  
For each `id` in the test set, predict up to 3 `Fertilizer Name` values, space-delimited.  
Example:
```
id,Fertilizer Name
750000,14-35-14 10-26-26 Urea
750001,14-35-14 10-26-26 Urea
```

**Dataset Features:**
- **Inputs:** temperature, humidity, soil moisture, soil type, crop type, nitrogen, phosphorus, potassium
- **Target:** Fertilizer Name

---

## Repository Structure

- `neural_network.py` — Main neural network class and training logic
- `fertilizer.ipynb` — Jupyter notebook for data preprocessing, training, and prediction
- `train.csv` / `test.csv` — Competition data files
- `README.md` — This file

---

## Approach

### Data Preprocessing

- **Encoding:**  
  - Categorical columns (`Fertilizer Name`, `Soil Type`, `Crop Type`) are mapped to integer indices for model compatibility.
- **Missing Data:**  
  - Checked for missing values (none found in the provided code).
- **Feature Selection:**  
  - The `id` column is dropped before training.
- **Distributions:**  
  - (Optional) Plots for feature distributions are included as commented code.

### Model

A custom neural network is implemented in `neural_network.py`:
- **Architecture:**
  - 5 fully connected layers with increasing and decreasing sizes
  - Batch normalization and dropout after each layer for regularization
  - ReLU activations
- **Training:**
  - Uses mini-batch gradient descent (`batch_size=64`)
  - Adam optimizer
  - Cross-entropy loss
  - Training/validation split (default: 80% train, 20% validation)
- **Metrics:**
  - Accuracy, precision, recall, and F1-score are computed on the validation set
- **Prediction:**
  - For each test sample, the top 3 predicted fertilizer indices are output as a space-separated string

### Submission

- The predicted indices are mapped back to fertilizer names for submission.
- The output is formatted as required by the competition.

---

## Usage

### 1. Install Requirements

```bash
pip install torch pandas scikit-learn matplotlib
```

### 2. Prepare Data

Download `train.csv` and `test.csv` from the Kaggle competition and place them in the project directory.

### 3. Run the Notebook

Open and run `fertilizer.ipynb` step by step.  
Key steps:
- Data loading and encoding
- Model training and evaluation
- Generating predictions
- Mapping predictions back to fertilizer names

### 4. Example Code Snippets

#### Data Encoding (from `fertilizer.ipynb`)
```python
fertilizer_to_index = {name: idx for idx, name in enumerate(train_data["Fertilizer Name"].unique())}
index_to_fertilizer = {idx: name for name, idx in fertilizer_to_index.items()}

soil_to_index = {name: idx for idx, name in enumerate(train_data["Soil Type"].unique())}
crop_to_index = {name: idx for idx, name in enumerate(train_data["Crop Type"].unique())}

train_data = train_data.replace({
    "Fertilizer Name": fertilizer_to_index,
    "Soil Type": soil_to_index,
    "Crop Type": crop_to_index
})
test_data = test_data.replace({
    "Soil Type": soil_to_index,
    "Crop Type": crop_to_index
})
train_data = train_data.drop(['id'], axis=1)
test_data = test_data.drop(['id'], axis=1)
```

#### Model Training and Prediction (from `fertilizer.ipynb`)
```python
import neural_network

nn_model = neural_network.nn_wrapper(train_data, test_data)
nn_model.train(plot=True)
nn_model.calculate_metrics()
predictions = nn_model.predict()
predictions = predictions.replace(index_to_fertilizer)
```

#### Submission Preparation
```python
submission = pd.read_csv('test.csv')[['id']]
submission['Fertilizer Name'] = predictions.values
submission.to_csv('submission.csv', index=False)
```

---

## Model Code Highlights (`neural_network.py`)

- **Batching:** Uses `TensorDataset` and `DataLoader` for efficient mini-batch training.
- **Device Support:** Automatically uses GPU if available.
- **Top-k Prediction:** Uses `torch.topk` to select the top 3 fertilizer predictions per sample.
- **Metrics:** Prints and returns accuracy, precision, recall, and F1-score.

#### Example: Top-k Prediction Formatting
```python
with torch.inference_mode():
    y_pred = self.model(self.test_data)
    _, topk_indices = torch.topk(y_pred, k=3, dim=1)
    topk_str = [' '.join(map(str, row)) for row in topk_indices.cpu().numpy()]
return pd.Series(topk_str, index=self.test_data_index, name="Prediction")
```

---

## Results

- The model outputs a space-separated string of the top 3 predicted fertilizers for each test sample, as required by the competition.
- Validation metrics are printed after training for model assessment.

---

## Acknowledgements

- [Kaggle Playground Series - Season 5, Episode 6](https://www.kaggle.com/competitions/playground-series-s5e6/)
- PyTorch, pandas, scikit-learn

---

## License

This project is provided for educational purposes. Please check the competition rules for allowed code.