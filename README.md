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

## Data Preprocessing
- It cleans the tweet texts by converting to lowercase, removing URLs, punctuation, mentions, and hashtags.
- Stopwords (common, uninformative words) are removed from the cleaned texts.
- The target sentiment labels are encoded into numeric values using label encoding.
- Unnecessary columns are dropped, keeping only the cleaned text and encoded labels.
- The cleaned texts are tokenized into sequences of word indices, limited to the top 50,000 words.
- These sequences are then padded to a fixed length of 100 tokens for uniform input size.
- Finally, the prepared input data (`train_data`, `test_data`) and corresponding numeric labels (`train_labels`, `test_labels`) are ready for model training and evaluation.

## Build the Word Embeddings using pretrained Word2vec/Glove (Text Representation)
- This code loads pre-trained GloVe word embeddings and creates an embedding matrix for the tokenizer's vocabulary:
- It reads the GloVe embeddings file (`glove.6B.200d.txt`), which contains word vectors of dimension 200, and loads these vectors into a dictionary mapping words to their embeddings.
- It prints how many word vectors were loaded.
- It initializes an embedding matrix of zeros with shape `(MAX_NB_WORDS, 200)`, where `MAX_NB_WORDS` is the maximum vocabulary size used during tokenization.
- For each word in the tokenizer’s word index (up to the maximum number of words), it looks up the corresponding GloVe embedding.
- If the embedding is found, it assigns the pre-trained vector to the respective row in the embedding matrix.
- This embedding matrix can then be used to initialize the embedding layer in a neural network, allowing the model to leverage pre-trained semantic word representations.

## Train the Deep Recurrent Model using RNN/LSTM 
At this stp, code builds, compiles, trains, and evaluates an LSTM-based neural network for sentiment classification:

- It creates a Sequential model with an embedding layer initialized with the pre-trained GloVe embeddings (non-trainable).
- Applies SpatialDropout1D to reduce overfitting by dropping entire 1D feature maps.
- Adds three stacked LSTM layers with dropout and recurrent dropout for capturing sequential dependencies in texts.
- The final Dense layer has 5 units with softmax activation to predict probabilities for 5 sentiment classes.
- The model uses sparse categorical cross-entropy loss and the Adam optimizer, tracking accuracy during training.
- EarlyStopping callback monitors validation loss to stop training if no improvement occurs for 3 consecutive epochs, restoring the best weights.
- The model is trained on training data with 20% reserved for validation.
- Finally, it evaluates and prints the training accuracy.

## Evaluation and model predictions on the test dataset 
- The test accuracy is 0.689, indicating 68.9% correct predictions on the test set.
- Prediction probabilities for each class with `model.predict(test_data)` are generated.
- `argmax`selects the class with the highest probability as the predicted class index.
- Prints the first 10 predicted and actual classes for comparison.
- You can further evaluate these predictions against true labels to calculate metrics like precision, recall, F1-score, and confusion matrix.

