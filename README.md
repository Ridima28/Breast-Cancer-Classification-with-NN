# Breast Cancer Classification with a Neural Network

Binary classification of breast tumors as **malignant** or **benign** from 30 numeric features computed from digitized images of fine-needle aspirates (FNA) of breast masses. The model is a small feed-forward neural network built with Keras.

> **Disclaimer:** This is a learning project. It is not a medical device and must not be used for diagnosis.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **95.80%** (143 held-out samples) |
| Test loss | 0.1280 |
| Final training accuracy | ~95.3% |
| Final validation accuracy | ~93.0% (10% of the training set) |

Training and validation accuracy/loss curves are plotted in the notebook.

## Dataset

- File: `data.csv` (Wisconsin Diagnostic Breast Cancer format), not included in this repo. Place it in the project root.
- 569 samples, 32 columns after cleaning (30 features + target).
- Class balance: 357 benign (B), 212 malignant (M).
- Features are the mean, standard error, and "worst" (largest) value of 10 cell-nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.
- No missing values. The `id` column and the empty `Unnamed: 32` column are dropped.

## Pipeline

1. **Load and inspect:** shape, dtypes, null check, summary statistics, class-wise feature means.
2. **Clean:** drop `id` and `Unnamed: 32`.
3. **Encode labels:** `LabelEncoder` on `diagnosis`.
4. **EDA:** class count plot, per-feature distributions and box plots, correlation heatmap.
5. **Split:** `train_test_split` with `random_state=42` and the default 75/25 split (426 train / 143 test).
6. **Scale:** `StandardScaler`, fitted on the training set only and applied to the test set.
7. **Model:**

   | Layer | Units | Activation |
   |---|---|---|
   | Flatten (input shape 30) | – | – |
   | Dense | 20 | ReLU |
   | Dense (output) | 2 | Sigmoid |

8. **Compile and train:** Adam optimizer, `sparse_categorical_crossentropy` loss, 10 epochs, `validation_split=0.1`, `tf.random.set_seed(3)`.
9. **Evaluate:** accuracy and loss on the test set, then predicted probabilities converted to labels with `argmax`.
10. **Predict on new input:** a single 30-value sample is reshaped, scaled with the fitted scaler, and passed to the model.

## Label encoding

`LabelEncoder` sorts classes alphabetically, so:

| Class | Encoded value |
|---|---|
| B (Benign) | 0 |
| M (Malignant) | 1 |

Use this mapping when interpreting `argmax` outputs.

## Setup

```bash
git clone https://github.com/Ridima28/Breast-Cancer-Classification-with-NN
cd Breast-Cancer-Classification-with-NN


pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

Put `data.csv` in the project root, then:

```bash
jupyter notebook main.ipynb
```

Run all cells top to bottom.

## Project structure

```
.
├── main.ipynb      # full workflow: EDA, preprocessing, training, evaluation, inference
├── data.csv        # dataset (add manually)
└── README.md
```

The correlation heatmap is also saved to disk as `Correlation Heat map` by the notebook.

## Possible improvements

- Report precision, recall, F1, and a confusion matrix. Recall on the malignant class matters most for this problem.
- Use a 1-unit sigmoid output with `binary_crossentropy`, or a 2-unit softmax output, instead of a 2-unit sigmoid.
- Replace `Flatten(input_shape=...)` with `keras.Input(shape=(30,))` to remove the Keras warning.
- Add early stopping and train for more epochs.
- Cross-validation and a comparison against baselines (logistic regression, random forest).
- Drop highly correlated features (radius, perimeter, and area are near-redundant) and check the effect.
- Save the fitted scaler and trained model for reuse.

## Tech stack

Python, NumPy, pandas, Matplotlib, seaborn, scikit-learn, TensorFlow/Keras.