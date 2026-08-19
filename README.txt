# Rain Prediction — Hybrid ML Model

Predicting whether it will rain using basic weather measurements — temperature,
humidity, wind speed, cloud cover, and pressure. Two classifiers are trained and
compared, then combined into a soft-voting hybrid model to see if blending them
improves on either one alone.

## Dataset

- 2,500 weather records
- Features: `Temperature`, `Humidity`, `Wind_Speed`, `Cloud_Cover`, `Pressure`
- Target: `Rain` (`rain` / `no rain`)
- No missing values
- Class imbalance: 2,186 "no rain" vs 314 "rain" — handled with SMOTE on the
  training set only

## Approach

1. **Preprocessing** — encode the target, split 80/20 (stratified), scale
   features with `StandardScaler`, oversample the minority class with SMOTE
   (train set only, to avoid leaking information into evaluation).
2. **Model 1 — Logistic Regression** — linear baseline.
3. **Model 2 — SVM (RBF kernel)** — captures non-linear relationships between
   the weather variables.
4. **Hybrid model** — soft voting: average the predicted probability of rain
   from both models, then apply a 0.5 threshold.
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, and
   confusion matrices for all three models.

## Results

| Model                 | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression    | 0.8940   | 0.5472    | 0.9206 | 0.6864   | 0.9640  |
| SVM (RBF)              | 0.9520   | 0.7294    | 0.9841 | 0.8378   | 0.9960  |
| Hybrid (SVM + LogReg)  | 0.9460   | 0.7000    | 1.0000 | 0.8235   | 0.9930  |

**Best model: SVM (RBF)** — highest F1-score and ROC-AUC of the three. The
dataset is imbalanced, so these two metrics are more trustworthy than raw
accuracy. The Hybrid model achieves perfect recall (never misses a rain event)
but at a small cost to precision and F1, since averaging in the weaker
Logistic Regression model pulled SVM's stronger individual performance down
slightly rather than improving it.

If the priority is catching every rain event, even at the cost of more false
alarms, the Hybrid model is the more defensible choice. Otherwise, SVM (RBF)
is the strongest overall performer and is used to generate the final
predictions.

## Project structure

```
├── Rain_Prediction_Cleaned.ipynb   # main notebook: preprocessing, models, evaluation
├── weather_forecast_data.csv       # dataset
├── final_predictions.csv           # output: actual vs predicted rain + probability
└── README.md
```

## Tech stack

Python, pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn

## Running it

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
jupyter notebook Rain_Prediction_Cleaned.ipynb
```
