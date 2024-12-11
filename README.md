### Overview
Neural network classifiers for classification of moles into benign and malignant categories.

### Models
We consider 3 models:
1. An initial CNN based classifier (InitialExploration.py and Model.py)
2. ResNet-101 (FineTuning.py)
3. A hyper-parameter optimized CNN classifier (see HyperparameterAnalysis.py and Model.py)

### Instructions for Execution
- For model 1, just run InitialExploration.py
- For the ResNet-101 model, just run FineTuning.py
- For the hyperparameter optimized classifier, first run HyperparameterOptimization.py, then run HyperparameterAnalysis.py

### Test Set Performance Summaries
#### Initial Implementation:
Test Accuracy: 0.85
Test Precision: 0.82
Test Recall: 0.88
Test F1 Score: 0.85

#### Transfer Model:
Test Accuracy: 0.84
Test Precision: 0.85
Test Recall: 0.81
Test F1 Score: 0.83

#### Optimized Hyper-parameters:
Test Accuracy: 0.45
Test Precision: 0.46
Test Recall: 1.00
Test F1 Score: 0.63

### Remarks
The performance summaries of our models indicate that hyper-parameter optimization possibly overfit the training set.