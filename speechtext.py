import pandas as pd
import numpy as np
import nltk
import pickle
import string
from nltk.corpus import stopwords
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pyttsx3
import torch.optim as optim


#speech recogniser
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech (words per minute)
engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)

def speak(text):
    engine.say(text)  # Convert the text to speech
    engine.runAndWait()  # Wait for the speech to finish playing

#nltk.download('stopwords')
def remove_punc(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return nopunc

def process_words(text):
    clean_words = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return clean_words

df1 = pd.read_csv('emails.csv')
x = df1.drop('Email No.', axis=1)
df= x.drop('Prediction', axis=1)
column_headers = df.columns.values.tolist()


# Sample variable
text1 = "under the hood, and it makes sense. The use of pickling conserves memory, enables start-and-stop model training, and makes trained models portable (and, thereby, shareable). Pickling is easy to implement, is built into Python without requiring additional dependencies, and supports serialization of custom objects. There’s little doubt about why choosing pickling for persistence is a popular practice among Python programmers and ML practitioners."
text = remove_punc(text1)

#print(text)
# List of words to count


# Count the occurrences of each word
word_counts = {word: text.count(word) for word in column_headers}
count_df = pd.DataFrame(word_counts, index=[0])
print(count_df)

# Display the word count

pickled_model = pickle.load(open('model.pkl', 'rb'))
print(pickled_model)
output = pickled_model.predict(count_df)

if count_df == 0:
    text_read = 'spam'
else:
    text_read = 'ham'