import nltk
import re
import numpy as np
import streamlit as st
import pickle

nltk.download('punkt')
nltk.download('stopwords')

# CLEAN RESUME FUNCTION

def clean_resume(text):
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r'http\S+\s', '', text)     # remove URLs
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    return text.strip()



# LOAD MODELS

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))



# STREAMLIT APP

def main():

    st.title("Resume Screening App")

    uploaded_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])

    if uploaded_file is not None:

        resume_bytes = uploaded_file.read()

      
        try:
            resume_text = resume_bytes.decode('utf-8')
        except:
            resume_text = resume_bytes.decode('latin-1', errors='ignore')
        cleaned_resume = clean_resume(resume_text)

        cleaned_tfidf = tfidf.transform([cleaned_resume])
        predictionID = clf.predict(cleaned_tfidf)

        st.write("Prediction:", predictionID)


if __name__ == '__main__':
    main()
