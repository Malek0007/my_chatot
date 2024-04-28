import random
import json
import pickle
import numpy as np
import tensorflow as tf
import os

import nltk
from nltk.stem import WordNetLemmatizer

# Set NLTK data path
nltk_data_path = r'C:\Users\dynabook\AppData\Roaming\nltk_data'  # Change this path if your NLTK data is stored elsewhere
os.environ['NLTK_DATA'] = nltk_data_path

# Download the 'punkt' tokenizer
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents data from JSON file
intents_path = r'C:\Users\dynabook\Desktop\chatbot\intents.json'  # Change this path to the location of your intents file
with open(intents_path, 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Preprocess intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize patterns
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        # Add tokenized pattern and intent tag to documents
        documents.append((wordList, intent['tag']))
        # Add intent tag to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove ignoreLetters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save preprocessed data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

# Create training data
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert training data to numpy array
random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')

print('Done')
