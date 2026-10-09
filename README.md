# Network Intrusion Detection Generalization Analysis

Evaluating how machine-learning intrusion detection models generalize across different CIC-IDS2017 traffic scenarios.

## Overview

This project compares Logistic Regression and Random Forest models for network intrusion detection using the CIC-IDS2017 dataset.

The main objective is to evaluate whether high performance on a conventional random train/test split actually translates into reliable detection on traffic from a different day.

## Experiments

### 1. Random Train/Test Split
Random Forest achieved approximately:
- Accuracy: 99.9%
- Attack Recall: 99.98%

### 2. Cross-Day Testing
Trained on Friday PortScan traffic and tested on Wednesday traffic.

Random Forest:
- Accuracy: 71.25%
- Attack Recall: 20.16%
- F1 Score: 33.56%

### 3. More Diverse Training Data
Training data was expanded using Friday PortScan and Thursday Web Attack traffic.

Random Forest improved to:
- Accuracy: 76.10%
- Attack Recall: 33.61%
- F1 Score: 50.32%

## Key Finding

Near-perfect performance on a random train/test split did not translate into strong performance on unseen traffic.

Adding more diverse training data improved cross-day detection, but the model still missed a significant proportion of attacks.

This demonstrates why intrusion detection models should be evaluated under realistic distribution shifts rather than relying only on benchmark accuracy.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- CIC-IDS2017
