# Otomoto ANN Optimization

BAN6440 Module 6 project by Emmanuel Fru.

## Project Overview

This project optimizes an artificial neural network for a marketing-related customer prediction task.

The assignment frames the case as Otomoto marketing segmentation. However, the supplied dataset is a telecommunications customer churn dataset rather than an automotive segmentation dataset. I therefore treated the problem as supervised churn classification. The marketing use case remains relevant because predicted churn risk can be used to support targeted retention campaigns.

## Dataset

The supplied `teleconnect.csv` dataset contains:

- 7,043 customer records
- 21 original columns
- 19 predictors after removing `customerID` and the target
- Binary target: `Churn`
- 5,174 non-churners
- 1,869 churners
- Majority-class baseline accuracy: 73.46%

Eleven `TotalCharges` values were blank. All occurred for customers with `tenure = 0`, so they were converted to `0` rather than dropping those customers.

## Legacy ANN

The supplied ANN workflow was recreated first to establish a baseline.

Legacy preprocessing and training:

- Label encoding for categorical variables
- 11 blank `TotalCharges` rows dropped
- 75/25 train-test split
- SMOTE applied to the training data
- No numeric scaling
- Adam optimizer
- 5 epochs
- Binary cross-entropy loss
- Test set used as validation during training

Legacy ANN architecture:

```text
19 inputs
-> Dense(19, ReLU)
-> Dense(15, ReLU)
-> Dense(10, ReLU)
-> Dense(1, Sigmoid)
```

Legacy results:

| Metric | Result |
|---|---:|
| Accuracy | 0.569 |
| Precision | 0.370 |
| Recall | 0.887 |
| F1-score | 0.522 |
| ROC-AUC | 0.744 |
| True negatives | 586 |
| False positives | 705 |
| False negatives | 53 |
| True positives | 414 |

The legacy model achieved high recall but generated too many false positives for an efficient retention campaign.

## Revised Preprocessing

The revised pipeline:

- retains all 7,043 records;
- converts blank `TotalCharges` values to `0`;
- removes `customerID`;
- encodes `Churn` as No = 0 and Yes = 1;
- uses one-hot encoding for categorical predictors;
- standardizes numeric predictors using `StandardScaler`;
- fits preprocessing only on the training split;
- uses a stratified 70/15/15 train-validation-test split;
- produces 30 encoded ANN inputs.

Final split:

| Split | Rows | Churn Rate |
|---|---:|---:|
| Train | 4,930 | 26.53% |
| Validation | 1,056 | 26.52% |
| Test | 1,057 | 26.58% |

## Revised ANN

The controlled optimizer experiment uses:

```text
30 encoded inputs
-> Dense(32, ReLU)
-> Dropout(0.30)
-> Dense(16, ReLU)
-> Dense(1, Sigmoid)
```

Common settings:

- Loss: binary cross-entropy
- Batch size: 32
- Maximum epochs: 150
- Early stopping: validation loss, patience = 10
- Classification threshold: 0.50

## Optimizer Experiment

Three optimizer configurations were compared:

- Adam, learning rate = 0.001
- RMSprop, learning rate = 0.001
- SGD, learning rate = 0.01

Each optimizer was trained across five random seeds:

```text
42, 123, 456, 789, 2026
```

The customer split remained fixed for all runs. Only model initialization and training randomness changed.

Optimizer selection was based only on validation results.

Selection rule:

1. Highest mean F1-score
2. Highest mean recall
3. Highest mean precision

### Validation Results

| Optimizer | Accuracy | Precision | Recall | F1 | ROC-AUC | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| Adam | 0.802 +/- 0.006 | 0.648 +/- 0.024 | 0.564 +/- 0.027 | 0.602 +/- 0.008 | 0.846 +/- 0.002 | 31.8 +/- 5.3 |
| RMSprop | 0.801 +/- 0.006 | 0.647 +/- 0.021 | 0.550 +/- 0.027 | 0.594 +/- 0.012 | 0.843 +/- 0.001 | 33.6 +/- 7.8 |
| SGD | 0.804 +/- 0.002 | 0.648 +/- 0.010 | 0.576 +/- 0.032 | 0.609 +/- 0.014 | 0.845 +/- 0.001 | 97.6 +/- 32.4 |

SGD was selected because it achieved the highest mean F1-score and recall under the predefined selection rule.

Its advantage over Adam was small, so the result should not be interpreted as evidence that SGD is universally better. Adam converged substantially faster and produced more stable F1 results.

## Final Held-Out Test Result

After optimizer selection, the final SGD configuration was trained using seed 42 and evaluated once on the held-out test set.

Final test results:

| Metric | Result |
|---|---:|
| Accuracy | 0.807 |
| Precision | 0.690 |
| Recall | 0.498 |
| F1-score | 0.579 |
| ROC-AUC | 0.845 |
| True negatives | 713 |
| False positives | 63 |
| False negatives | 141 |
| True positives | 140 |

Majority-class test accuracy was 0.734, so the ANN improved accuracy by approximately 7.3 percentage points.

The final model is more selective than the legacy model. Precision, F1-score, accuracy, and ROC-AUC improved, while recall decreased.

The final SGD run reached the 150-epoch limit and achieved its best validation loss at epoch 149. This suggests that convergence should be revisited in future validation-only experiments.

## Project Structure

```text
otomoto-ann-optimization/
|
|-- data/
|   `-- teleconnect.csv
|
|-- src/
|   |-- data_loader.py
|   |-- legacy_baseline.py
|   |-- preprocessing.py
|   |-- preprocessing_check.py
|   |-- model.py
|   |-- evaluation.py
|   |-- optimizer_experiment.py
|   |-- analyze_optimizer_results.py
|   `-- final_evaluation.py
|
|-- tests/
|   `-- test_pipeline.py
|
|-- outputs/
|   |-- experiment_results.csv
|   |-- experiment_histories.csv
|   |-- optimizer_summary.csv
|   |-- optimizer_summary_detailed.csv
|   |-- optimizer_selection.txt
|   |-- optimizer_f1_comparison.png
|   |-- optimizer_epochs_comparison.png
|   |-- optimizer_loss_comparison.png
|   |-- final_metrics.txt
|   |-- final_confusion_matrix.png
|   |-- final_roc_curve.png
|   `-- pytest_results.txt
|
|-- module_6_otomoto_ann_optimization_report.tex
|-- requirements.txt
`-- README.md
```

## Environment Setup

From the project directory in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Expected main packages:

```text
pandas
numpy
scikit-learn
imbalanced-learn
tensorflow
matplotlib
pytest
```

## Run the Project

### 1. Check preprocessing

```powershell
.\.venv\Scripts\python.exe src\preprocessing_check.py
```

### 2. Recreate the legacy baseline

```powershell
.\.venv\Scripts\python.exe src\legacy_baseline.py
```

### 3. Run the optimizer experiment

```powershell
.\.venv\Scripts\python.exe src\optimizer_experiment.py
```

This trains Adam, RMSprop, and SGD across five random seeds.

### 4. Analyze optimizer results

```powershell
.\.venv\Scripts\python.exe src\analyze_optimizer_results.py
```

### 5. Run final evaluation

```powershell
.\.venv\Scripts\python.exe src\final_evaluation.py
```

The final test set should only be evaluated after optimizer selection is locked.

Do not retune the model using the final test results.

### 6. Run automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Final test status:

```text
11 passed in 6.45s
```

To save the test log:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v |
Tee-Object -FilePath .\outputs\pytest_results.txt
```

## Report

The report is written in LaTeX:

```text
module_6_otomoto_ann_optimization_report.tex
```

Compile with:

```powershell
pdflatex module_6_otomoto_ann_optimization_report.tex
pdflatex module_6_otomoto_ann_optimization_report.tex
```

Running `pdflatex` twice resolves table, figure, and citation references.

## Methodological Notes

The large performance difference between the legacy ANN and the final model should not be attributed to optimizer choice alone.

The revised workflow also changed:

- categorical encoding;
- scaling;
- missing-value handling;
- data splitting;
- validation isolation;
- ANN architecture;
- training duration.

The controlled evidence for optimizer choice is the five-seed comparison performed within the revised pipeline.

The test set was held out during optimizer comparison and evaluated only after SGD was selected.

## Limitations

Main limitations include:

- only one fixed train-validation-test split was used;
- the optimizer comparison used five seeds but no repeated cross-validation;
- the default 0.50 classification threshold was not optimized for business cost;
- final test recall was 49.8%;
- the final SGD run reached the 150-epoch cap;
- the supplied dataset did not match the Otomoto automotive framing in the assignment.

## AI Use

AI tools were used to support research, implementation, debugging, experiment design, result interpretation, report structuring, and review.

The code was executed locally, outputs were inspected, results were checked against saved files, and model decisions were based on the observed results rather than copied directly from AI output.

The final assignment should include the required AI Disclosure Form describing the tools used, how prompts were refined, what was verified, and what was changed after review.

## Author

**Emmanuel Fru**  
BAN6440  
Nexford University
