# Amazon Product Reviews Sentiment Analysis

This project performs sentiment analysis on Amazon product reviews using various approaches, including lexicon-based, aspect-based, and machine learning methods such as SVM, Naive Bayes, and RNN. The dataset used is `Reviews.csv` from Amazon product reviews.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Sentiment Analysis Methods](#sentiment-analysis-methods)
- [Results](#results)
- [License](#license)

## Project Overview
This project implements five different sentiment analysis approaches to classify reviews as positive or negative. The sentiment analysis methods include:

1. **Lexicon-Based Sentiment Analysis**
2. **Aspect-Based Sentiment Analysis**
3. **Naive Bayes Sentiment Analysis**
4. **Support Vector Machine (SVM) Sentiment Analysis**
5. **Recurrent Neural Network (RNN) Sentiment Analysis**

## Dataset
The dataset used in this project is the `Reviews.csv` file, which contains product reviews and ratings from Amazon.

You can access the dataset from Kaggle at the following link:
- [Kaggle: Amazon Reviews Dataset](https://www.kaggle.com/code/haidytalaat/sentimentent-analysis-modeling-on-amazon-reviews/input?select=Reviews.csv)

### Dataset Structure:
- `Text`: The actual review text.
- `Score`: The rating of the product (1-5 stars).
- `Sentiment`: Binary sentiment label (1 for positive, 0 for negative). The sentiment is derived from the `Score` (reviews with 4-5 stars are labeled positive, 1-3 stars are negative).

## Installation
Clone this repository to your local machine and install the required dependencies:

```bash
git clone https://github.com/LeylaY1996/sentiment-analysis-aws-reviews.git
cd sentiment-analysis-aws-reviews
pip install -r requirements.txt
