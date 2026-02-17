import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("AI Fake News Detection System")

st.write("Enter news text below:")

input_text = st.text_area("News Text")

if st.button("Predict"):

    if input_text.strip() == "":
        st.warning("Please enter text")

    else:
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)

        if prediction[0] == 1:
            st.success("This is REAL News")
        else:
            st.error("This is FAKE News")
