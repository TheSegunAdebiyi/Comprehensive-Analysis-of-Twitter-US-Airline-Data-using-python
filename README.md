# Social Media Sentiment Analysis Project

## Overview
This project aims to analyze social media sentiment using the Twitter US Airline Sentiment dataset. We explore trends, perform sentiment analysis, and build a simple machine learning model for predicting sentiment based on tweet text.In the era of digital communication, social media platforms play a crucial role in shaping public opinion. This project delves into the realm of social media sentiment analysis, using the Twitter US Airline Sentiment dataset to uncover insights into user sentiments towards airline services.
In the realm of social media, Twitter stands as a powerful platform for users to voice their opinions and experiences. This project embarks on a data-driven journey, exploring sentiments expressed on Twitter towards US airlines. Leveraging the Twitter US Airline Sentiment dataset from Kaggle, we aim to uncover insights, patterns, and sentiments that encapsulate the public's perception of airline services.


## Project Structure
1. **Data Collection and Dataset Overview
:**
   - Utilized the Twitter US Airline Sentiment dataset from Kaggle.
   - Collected relevant data using Python libraries such as Tweepy, Instaloader, or PRAW

The Twitter US Airline Sentiment dataset, sourced from Kaggle, comprises 14,640 tweets. Before diving into the analysis, let's take a look at some key statistics
```python
print(df.columns)
selected_columns = ['tweet_id', 'airline_sentiment', 'text', 'tweet_created', 'retweet_count']
df_selected = df[selected_columns]
print(df_selected.head())

```

2. **Data Cleaning and Preprocessing:**
   - Cleaned the dataset by handling missing values and converting data types.
   - Converted the 'tweet_created' column to datetime format.

Before delving into analysis, it's imperative to ensure the dataset is clean and ready for exploration. Data cleaning involves handling missing values, converting data types, and addressing any anomalies.

  
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

df_selected = df[selected_columns].copy()
df_selected['tweet_created'] = pd.to_datetime(df_selected['tweet_created'])

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('Tweets.csv')


# Select relevant columns for analysis
selected_columns = ['tweet_id', 'airline_sentiment', 'text', 'tweet_created', 'retweet_count']
df_selected = df[selected_columns]


# Convert 'tweet_created' to datetime format
df_selected['tweet_created'] = pd.to_datetime(df_selected['tweet_created'])
```

3. **Exploratory Data Analysis (EDA):**
   - Explored basic statistics and visualizations to understand the dataset.
   - Visualized sentiment distribution and posting patterns over time.

#### Sentiment Distribution
Understanding the sentiment distribution is fundamental to grasping the overall tone of the dataset. Visualizing the count of each sentiment class provides a quick snapshot


4. **Sentiment Analysis:**
   - Utilized TextBlob for sentiment analysis on tweet text.
   - Visualized sentiment polarity distribution.
```python
# Display basic statistics
print(df_selected.describe())

# Visualize sentiment distribution
sns.countplot(x='airline_sentiment', data=df_selected)
plt.title('Sentiment Distribution')
plt.show()

```
5. **Time Series Analysis:**
   - Analyzed posting patterns over time using time series analysis.
   - Resampled data by day and visualized daily total retweet count.
     
#### Retweet Patterns
Analyzing retweet patterns can uncover the tweets that resonate most with users. Let's explore the highest retweet occurrence and its impact on the overall engagement.
```python
# Set 'tweet_created' as the index for time series analysis
df_selected.set_index('tweet_created', inplace=True)

# Resample data by day and calculate the sum of retweet_count
resampled_data = df_selected['retweet_count'].resample('D').sum()

# Plot the time series
plt.figure(figsize=(12, 6))
resampled_data.plot()
plt.title('Daily Total Retweet Count Over Time')
plt.xlabel('Date')
plt.ylabel('Total Retweet Count')
plt.show()
```

### Sentiment Polarity
Delving into sentiment polarity, we explore the sentiments present in the dataset and visualize their distribution.
```python
pip install textblob
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


from textblob import TextBlob

# Function to get sentiment polarity
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to the 'text' column
df_selected['sentiment_polarity'] = df_selected['text'].apply(get_sentiment)

# Visualize sentiment polarity distribution
sns.histplot(x='sentiment_polarity', data=df_selected, bins=50, kde=True)
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Sentiment Polarity')
plt.show()
```


6. **Machine Learning Model:**
   - Prepared data by converting text to numerical features using TF-IDF vectorization.
   - Trained a Multinomial Naive Bayes classifier for sentiment prediction.

#### Feature Engineering
The transition to machine learning involves converting text data into a format suitable for model training. We employ TF-IDF vectorization to transform tweet text into numerical features.

```python
pip install pandas matplotlib seaborn textblob scikit-learn



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('Tweets.csv')

# Select relevant columns for analysis
selected_columns = ['airline_sentiment', 'text']
df_selected = df[selected_columns]

# Convert 'airline_sentiment' to numerical labels
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df_selected['sentiment_label'] = df_selected['airline_sentiment'].map(sentiment_mapping)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_selected['text'], df_selected['sentiment_label'], test_size=0.2, random_state=42)


df_selected = df[selected_columns].copy()
df_selected['sentiment_label'] = df_selected['airline_sentiment'].map(sentiment_mapping)


# Use TF-IDF vectorization for text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)



# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)


```


7. **Evaluation:**
   - Evaluated the machine learning model using classification report, accuracy score, and confusion matrix.
   - Provided insights into model performance and potential areas for improvement.

A Multinomial Naive Bayes classifier is chosen for sentiment prediction. The model is trained and evaluated using standard classification metrics.
```python
# Predictions on the test set
y_pred = nb_classifier.predict(X_test_tfidf)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sentiment_mapping.keys(), yticklabels=sentiment_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

8. **Conclusion:**
   - Summarized key findings, insights, and contributions of the project.
   - Discussed potential future enhancements or directions.

## Libraries Used
- pandas
- matplotlib
- seaborn
- textblob
- scikit-learn

## Instructions for Replication
1. Clone this repository.
2. Install required libraries using `pip install -r requirements.txt`.
3. Download the Twitter US Airline Sentiment dataset from Kaggle.
4. Run the Jupyter Notebook or Python script for step-by-step execution.

## Results
- Foundings from EDA, sentiment analysis, and time series analysis.
- Machine learning model performance metrics and insights.
  
#### Results and Insights
##### Sentiment Analysis Summary
Summarizing the sentiment analysis reveals intriguing insights. Negative sentiment dominates, suggesting potential areas for improvement in airline services.
Machine Learning Model Performance
The machine learning model achieves an accuracy of 73.8%, with distinct performance metrics for each sentiment class.

#### Visualizing the Journey
#### Highest Retweet Occurrence
A visual representation of the highest retweet occurrence on February 23rd offers a snapshot of user engagement peaks.

### Word Clouds
Word clouds visually represent the most frequent words associated with each sentiment, providing a qualitative view of user sentiments.

## Conclusion
In conclusion, this project has provided a comprehensive analysis of social media sentiment towards airline services. From sentiment distribution to machine learning model performance, the journey has uncovered valuable insights. The visualization of retweet patterns and sentiment polarity adds depth to the analysis.


### Future Directions
To enhance this analysis, future directions could involve:
-Exploring additional features for sentiment analysis.
-Experimenting with advanced machine learning models.
-Incorporating user feedback sentiment for a more holistic view.


## Acknowledgments
This project wouldn't be possible without the valuable Twitter US Airline Sentiment dataset from Kaggle and the open-source Python libraries utilized.

- Kaggle for providing the Twitter US Airline Sentiment dataset.
- Open-source Python libraries used in the project.

## Author
[Adebiyi Segun Timothy]

