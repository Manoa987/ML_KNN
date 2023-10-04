# Sentiment Analysis on User Reviews

## Dataset Description

The dataset `reviews_sentiment.csv` contains 257 records with user opinions about an application. It consists of several variables:

- `Review Title`: The title of the review.
- `Review Text`: The actual review text.
- `wordcount`: The word count in the review text.
- `Title sentiment`: Estimated sentiment in positive (assigned 1) or negative (assigned 0). May contain NaN values.
- `text sentiment`: Positive or negative sentiment provided by the reviewer.
- `sentimentValue`: A real value ranging from -4 to 4 indicating the positivity or negativity of the review.
- `Star Rating`: User ratings ranging from 1 to 5.

## Tasks

### Word Count Analysis

1. Determine the average word count for reviews rated with 1 star.

### Data Splitting

2. Split the dataset into a training set and a test set.

### K-Nearest Neighbors (K-NN) Classification

3. Implement the K-Nearest Neighbors (K-NN) algorithm for classifying reviews. Use the `Stars Rating` as the target variable and the numeric variables `wordcount`, `Title sentiment`, `sentimentValue` as explanatory variables. Experiment with different values of `k`.

### Weighted K-Nearest Neighbors Classification

4. Apply the K-NN algorithm with weighted distances to classify reviews, utilizing the same variables as in step 3. Experiment with different values of `k`.

### Accuracy and Confusion Matrix

5. Calculate the classifier's accuracy and construct the confusion matrix to evaluate the classification performance.

## Usage

1. Ensure you have the necessary Python libraries installed.
2. Execute the code files for word count analysis, data splitting, K-NN classification, weighted K-NN classification, accuracy calculation, and confusion matrix generation.

## Contributions

Contributions to this project are welcomed. If you have any improvements or suggestions, please contribute by opening issues or creating pull requests.
