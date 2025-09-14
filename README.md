# Covid-19-tweets-classification-of-texts

## Dataset 
The given challenge is to build a multiclass classification model to predict the sentiment of Covid-19 tweets. The tweets have been pulled from Twitter and manual tagging has been done. Information like Location, Tweet At, Original Tweet, and Sentiment are available.

The training dataset consists of 36000 tweets and the testing dataset consists of 8955 tweets. There are 5 sentiments namely ‘Positive’, ‘Extremely Positive’, ‘Negative’, ‘Extremely Negative’, and ‘Neutral’ in the sentiment column.

## ⚙️ Technologies 
- TensorFlow / Keras → Model building
- scikit-learn (sklearn) → Evaluation metrics, preprocessing
- Pandas / NumPy → Data handling & preprocessing
- Seaborn / Matplotlib → Data visualization
- WordCloud → Text visualization

## 📊 Exploratory Data Analysis (EDA)
Class Distribution:
- Positive → 27.7%
- Negative → 24.4%
- Neutral → 18.4%
- Extremely Positive → 16%
- Extremely Negative → 13.5%
The distribution was visualized using bar graphs, and pie charts clearly showing positive sentiment as the largest share.

## ☁️ Word Clouds

Word clouds were generated for each sentiment class. Some commonly observed frequent words across sentiments include:
“coronavirus”, “covid”, “supermarket”, “grocery store”, “food”, “price”, “https”
These words highlight the recurring themes of health concerns, shopping essentials, and online information sharing during the pandemic.
