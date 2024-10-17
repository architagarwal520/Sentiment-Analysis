# Sentiment Analysis of Patient Satisfaction Comments

This project applies machine learning models to analyze and classify sentiment in patient satisfaction comments. The objective is to understand the patients' experiences and sentiments based on their healthcare, with the aim to improve these experiences through better data-driven insights.

## Table of Contents
- Project Overview
- Dataset
- Models Used
- Feature Selection
- Evaluation
- How to Run
- Results
- Conclusion
- Future Work

## Project Overview
Sentiment analysis of patient satisfaction comments is an important problem in healthcare, as understanding patients' experiences can help improve service quality. In this project, we perform sentiment analysis using a dataset of patient comments. Each comment is labeled as positive or negative based on the patient's satisfaction with their healthcare experience.

The project compares two machine learning models: Naive Bayes and Logistic Regression. These models are evaluated based on their ability to predict sentiment correctly and their efficiency in handling large amounts of text data.

### Why Naive Bayes and Logistic Regression?
Naive Bayes is used because of its simplicity and effectiveness in text classification tasks. It models the probability of words or features occurring in each class and is efficient in handling sparse data like text.
Logistic Regression is employed for its ability to provide a linear relationship between features and output. It is interpretable, allows for regularization to avoid overfitting, and provides insights into which features contribute most to the prediction.

## Feature Selection
Feature selection is a key step in improving the model's efficiency. In this project, techniques such as chi-square and mutual information were used to reduce the number of features while maintaining model performance.

## Dataset
The dataset used in this project consists of patient reviews derived from ratemds.com, where each review is labeled with:

- Sentiment: The target variable, with values -1 (negative sentiment) and 1 (positive sentiment).
- Clinician Gender: Gender information for the clinician (0 = female, 1 = male, 2 = unknown).
- Review Text: The text of the review itself.

### Feature Representation
The comments were transformed into two main feature representations:

- TF-IDF: A 500-dimensional vector of the highest TF-IDF values from the review text.
- Word Embeddings: A 384-dimensional embedding generated using a pre-trained language model.

### Models Used

1. Naive Bayes:
Naive Bayes models were applied to capture the likelihood of features occurring in each class:
- Gaussian Naive Bayes (used for continuous features)
- Bernoulli Naive Bayes (used for binary features)
- Multinomial Naive Bayes (used for count-based features like TF-IDF)
2. Logistic Regression:
Logistic Regression was used to model the linear relationship between features and the sentiment. Regularization techniques (like L2) were applied to prevent overfitting.

### Feature Selection

The dataset contains high-dimensional data, and feature selection was critical in optimizing model performance. The following approaches were used:

- Chi-Square Test: Selected the top features based on their relationship with the target variable.
- Mutual Information: Measures the amount of information one feature provides about the target variable.
Feature selection helped reduce complexity without compromising model accuracy. Experiments showed that reducing features down to 250 using these techniques maintained the model's accuracy.

### Evaluation

The models were evaluated using several metrics:

- Accuracy: Percentage of correctly classified reviews.
- Precision, Recall, and F1-Score: Further analyzed model performance in handling class imbalance (positive and negative reviews).
- Macro and Weighted Averages: Used to better evaluate the model across imbalanced classes.
- The evaluation was conducted on both TF-IDF vectors and word embeddings.
