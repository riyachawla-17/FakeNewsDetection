import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_message(text):
    # change case to lowercase
    text = text.lower()
    # for tokenizing the message
    text = nltk.word_tokenize(text)

    # for removing alpha umeric values
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # for removing stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # for stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfdif = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('finalized_model.pkl','rb'))
st.title("Fake News Detection - Made By (19DIT006 & 19DIT013)")
input_text = st.text_input("Enter the News content")
if st.button("Detect"):
    # 1 Preprocess
    transform_text = transform_message(input_text)
    # 2 Vectorize
    vector_input = tfdif.transform([transform_text])
    # 3 Predict
    result = model.predict(vector_input)[0]
    # 4 Display
    if result == 1:
        st.header("Fake News")
    else:
        st.header("Real News")



