import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st 


df_real = pd.read_csv('True.csv',encoding='latin1',on_bad_lines='skip')
df_fake = pd.read_csv('Fake.csv',encoding='latin1',on_bad_lines='skip')

df_real['Label'] = 1
df_fake['Label'] = 0


df = pd.concat([df_real, df_fake])
df = df.sample(frac=1).reset_index(drop=True)

df.drop_duplicates(inplace=True)

new_df = df[['title','subject','Label']]

new_df['complete_text'] = new_df['title'] + " " + new_df['subject']

def clean_text(text):
    # Changing text to lower case
    text_clean = text.lower()
    # Removing unwanted characters but keeping spaces and words
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text_clean)
    # Tokenizing the words
    tokenize = word_tokenize(text_clean)
    # Removing stopwords and stemming the words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stemming = [stemmer.stem(word) for word in tokenize if word not in stop_words]
    cleaned_text = " ".join(stemming)
    
    return cleaned_text


new_df['clean_text'] = new_df['complete_text'].apply(clean_text)

X = new_df['clean_text'].values
y = new_df['Label'].values

# Fit the vectorizer only on the training data
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(X)

# Split the data into training and test sets before vectorizing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LogisticRegression()
model.fit(X_train,y_train)


# website
st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vectorizer.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Real')
    else:
        st.write('The News Is Fake')