# Credit Card Fraud Detection 💳🕵️‍♂️

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

A clean, end-to-end Machine Learning project demonstrating how to detect credit card fraud using Python. This project focuses on handling highly imbalanced datasets and evaluating models using real-world metrics (Precision, Recall, F1-Score) instead of relying on misleading accuracy scores.

## 🌟 Overview

Credit card fraud detection is a classic problem in machine learning. Because fraudulent transactions are extremely rare compared to normal transactions, the data is highly imbalanced. This project walks through a complete pipeline to solve this:
1. **Data Loading & Exploration**: Understanding the class imbalance.
2. **Data Balancing**: Utilizing **Undersampling** to create a balanced dataset.
3. **Data Preprocessing**: Scaling features using `StandardScaler`.
4. **Model Training**: Training both **Logistic Regression** and **Random Forest** classifiers.
5. **Evaluation**: Comparing models using classification reports and visualizing results with a Confusion Matrix.

## 🚀 Features

- **Real-World Imbalanced Data**: Handles data where frauds represent a tiny fraction of total transactions.
- **Industry-Standard Evaluation**: Uses Precision, Recall, and F1-Scores.
- **Multiple Models**: Compares a linear model (Logistic Regression) against an ensemble tree model (Random Forest).
- **Clean, Modular Code**: Written as a standard Python script (VS Code friendly, no Jupyter notebook dependency required).

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Download the Dataset
Due to its size, the dataset is not included in this repository.
1. Download the dataset from [Kaggle's Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Extract the downloaded archive.
3. Place the `creditcard.csv` file in the root directory of this project.

### 3. Install Dependencies
It is recommended to use a virtual environment.
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 💻 Usage

Run the main script to execute the entire pipeline (data loading, balancing, training, and evaluation):

```bash
python fraud_detection.py
```

### What to expect:
- The script will print the distribution of the original and balanced datasets.
- It will train both the Logistic Regression and Random Forest models.
- Text-based classification reports will be printed to your terminal.
- Finally, a visual **Confusion Matrix** for the Random Forest model will pop up in a new window.

## 🎯 Who this is for
- **ML Beginners & Students**: A great template for learning classification.
- **Portfolio Builders**: A clean example of handling imbalanced data to showcase on your resume.
- **Data Enthusiasts**: Anyone wanting to learn how to properly evaluate models beyond simple "accuracy".

## 📝 License
This project is open-source and available under the MIT License.
