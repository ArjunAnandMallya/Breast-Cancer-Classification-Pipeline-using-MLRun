# Breast Cancer Classification using MLRun  
**CS 203: Software Tools & Techniques for AI**  
**IIT Gandhinagar – Semester II, 2024–25**  


## Contributors
- Arjun Anand Mallya (23110039)  
- Venkatakrishnan E (23110357)

---

## Objective  
This lab demonstrates CI/CD for Machine Learning using MLRun. It involves building an ML pipeline for training, tuning, and deploying a Random Forest classifier on the breast cancer dataset using MLRun’s orchestration features.

---

## Dataset  
- Source: `sklearn.datasets.load_breast_cancer`  
- Binary classification problem (malignant vs benign)

---

## Files and Description

### `data_prep.py`
- Loads and preprocesses the breast cancer dataset.
- Returns an MLRun artifact (DataFrame).

### `trainer.py`
- Splits data into 90% train, 10% test.
- Trains a Random Forest Classifier.
- Uses `apply_mlrun()` to track and wrap the model.
- Outputs:
  - Trained model
  - Confusion matrix
  - ROC curve
  - Calibration curve
  - Feature importance plot

### `serving.py`
- Defines `ClassifierModel`, which inherits from `mlrun.serving.V2ModelServer`.
- Implements `load()` and `predict()` for model serving.

### `workflow.py`
- Uses `@dsl.pipeline` to define:
  - Data ingestion
  - Hyperparameter tuning: `n_estimators = [10, 100, 200]`, `max_depth = [2, 5, 10]`
  - Model selection based on maximum accuracy
  - Deployment using `mlrun.deploy_function`

---

## Visual Artifacts (Captured as Screenshots)
- Workflow Graph
- Data_prep artifact (DataFrame preview)
- Confusion Matrix
- Feature Importance plot
- ROC and Calibration Curves

---

## Execution
- Create MLRun project using:
- Run the workflow:

---

## Result
- Best model automatically selected and deployed
- Model served using MLRun’s serving runtime

## Learnings
- Automated ML pipelines using MLRun
- CI/CD in ML with orchestration and deployment
- Serving models with `V2ModelServer`
- Model tracking and reproducibility with MLRun artifacts
