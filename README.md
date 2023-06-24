# Machine Learning Email Spam Filter
This project presents a Machine Learning-based Email Spam Filter developed in Python. The aim is to correctly classify emails as either "Spam" or "Not Spam" based on various features extracted from the email texts. This is achieved by analyzing and extracting various features from the email text data, converting it into meaningful insights that the model can interpret. To accomplish this, the project follows a multi-stage process that begins with rigorous data preprocessing and cleaning. With the use of advanced Natural Language Processing (NLP) techniques, the email data is thoroughly processed to remove any unnecessary noise, and is subsequently tokenized, stemmed, and lemmatized to break down the text into more manageable and meaningful components.

## Features
- Preprocessing and cleaning of email dataset.
- Natural Language Processing (NLP) for text data manipulation.
- Feature extraction from preprocessed data.
- Training of multiple machine learning models for email classification.
- Evaluation and optimization of models.
- User-friendly interface for email spam prediction.

## Getting Started

### Prerequisites

- Python 3.7 or newer
- Pandas
- Numpy
- NLTK
- Scikit-Learn
- Tkinter

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the libraries. Example:
pip install pandas numpy nltk scikit-learn

### Installation

Clone the repository to your local machine:
git clone https://github.com/yourusername/ML_Email_Spam_Filter.git

## Usage

Run main.py to start the application. Enter the text of an email to predict whether it's spam or not.

## How it works

1. **Data Preprocessing and NLP**: The project begins with the preprocessing and cleaning of the email dataset. This stage is crucial to any data analysis and Machine Learning application, and here it involves handling missing values and outliers. For text data, Natural Language Processing (NLP) techniques are used to manipulate and convert the data into a format that can be utilized in the next stages. The text data is tokenized (splitting text into individual words), stemmed (reducing words to their root form), and lemmatized (applying a linguistic approach to reducing words to their root form). 
2. **Feature Extraction**: Features are extracted from the preprocessed data. In the context of this project, 'features' refer to the individual measurable properties or characteristics of the phenomena being observed, which in this case are the email texts. A Bag-of-Words (BoW) model is used to convert the email texts into numerical feature vectors. The BoW model represents text data in terms of the frequency of each word, disregarding the order in which they appear. The result of this process is a matrix where each row represents an email and each column represents a unique word in the email text. The value in each cell is the frequency of that word in the corresponding email.
3. **Model Training**: Several Machine Learning models are trained on the dataset. The models used in this project include Logistic Regression, Decision Trees, and Support Vector Machines (SVM).
4. **Model Evaluation and Optimization**: The models are then evaluated and compared to find the best performing one. Evaluation metrics such as precision, recall, F1 score, and ROC AUC score are used. These metrics provide different perspectives on the model's performance, such as its accuracy, its balance between sensitivity and specificity, and its performance across various threshold settings. To further optimize the model, techniques such as cross-validation and hyperparameter tuning are used.
5. **User Interface**: Lastly, a user-friendly interface is created using the Tkinter library, which allows users to input an email and receive a prediction on whether it's spam or not. This makes the model accessible to users who don't have a background in coding or data science.
