import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CFPB Complaint Classifier",
    page_icon="",
    layout="centered"
)

# --- 2. CLEANING FUNCTIONS (Copied from Notebook) ---
# We must redefine these functions to process user input 
# exactly as we processed the training data.

# Silent download of necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Same cleaning function as used in training pipeline.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove masks (XXXX) and non-alphabetical characters
    text = re.sub(r'X+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    # Tokenization and Lemmatization
    tokens = text.split()
    clean_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 2
    ]
    
    # Truncate to first 500 words (matching training constraints)
    clean_tokens = clean_tokens[:500]
    
    return " ".join(clean_tokens)

# --- 3. LOAD MODEL (Cached for performance) ---
import os

@st.cache_resource
def load_model():
    # Get the absolute path of the current file (streamlit_app.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct path to the parent directory, then to models
    model_path = os.path.join(current_dir, "..", "models", "nlp_classifier.pkl")
    
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Critical Error: Model file not found at path: {model_path}")
        return None

model = load_model()

# --- 4. USER INTERFACE ---
st.title("🏦 AI Complaint Classifier")
st.markdown("""
This model automatically analyzes financial complaints to route them to the correct department.
**Supported Categories:** *Mortgage, Credit Card, Debt Collection, Credit Reporting, Student Loan...*
""")

# Text Input Area
user_input = st.text_area("Type your complaint here:", height=150, placeholder="Ex: I have a problem with my credit report...")

if st.button("Analyze Complaint"):
    if user_input:
        if model:
            # A. Preprocessing
            cleaned_input = clean_text(user_input)
            
            # B. Prediction (Model expects a list)
            prediction = model.predict([cleaned_input])[0]
            probabilities = model.predict_proba([cleaned_input])[0]
            
            # C. Display Result
            st.success(f"Predicted Category: **{prediction}**")
            
            # D. Display Confidence Levels
            st.markdown("### Confidence Levels")
            
            # Create a dictionary mapping class names to probabilities
            class_probs = dict(zip(model.classes_, probabilities))
            # Sort by highest probability
            sorted_probs = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
            
            # Display top 3 categories
            for label, score in sorted_probs[:3]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(score)
                with col2:
                    st.write(f"{label} ({score:.1%})")
                    
            # Debugging section (Optional)
            with st.expander("See cleaned text (Debug)"):
                st.text(cleaned_input)
                
    else:
        st.warning("Please enter some text to analyze.")