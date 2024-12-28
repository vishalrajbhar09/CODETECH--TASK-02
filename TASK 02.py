#!/usr/bin/env python
# coding: utf-8

# # TASK TWO:ANALYSIS ON MOVIE REVIEWS
# 
# Develop a sentiment analysis model to classify movie reviews as positive or
# negative. Use a dataset like the IMDb Movie Reviews dataset for training and
# testing.

# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load IMDb dataset from keras
print("Loading IMDb dataset...")
num_words = 10000  # Use the top 10,000 most frequent words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to ensure uniform input size
maxlen = 200
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Define a simple neural network for sentiment classification
print("Building model...")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_words, output_dim=32, input_length=maxlen),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training model...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
print("Evaluating model...")
y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display some example reviews with predictions
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

print("\nExample Reviews with Predictions:")
for i in range(5):
    print(f"Review: {decode_review(X_test[i])}")
    print(f"Predicted Sentiment: {'Positive' if y_pred[i] == 1 else 'Negative'}\n")

      


# In[ ]:




