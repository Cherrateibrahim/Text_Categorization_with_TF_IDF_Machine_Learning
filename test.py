import streamlit as st
import joblib
import spacy
import json
from spacy.language import Language
from sklearn.feature_extraction.text import TfidfVectorizer

# Load label mapping from a JSON file (e.g., { "0": "business", "1": "entertainment", ... })
def load_label_mapping(json_path="label_mapping.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Class that handles model loading, preprocessing, and prediction
class ModelCategorization:
    def __init__(self, model_name, vectorizer_path, label_mapping_path):
        # Load the trained classification model
        self.model = joblib.load(model_name)
        # Load the vectorizer (TF-IDF)
        self.vectorizer = joblib.load(vectorizer_path)
        # Load the spaCy language model for English
        self.nlp: Language = spacy.load("en_core_web_sm")
        # Get stop words
        self.stop_words = self.nlp.Defaults.stop_words
        # Load the label mapping
        self.label_mapping = load_label_mapping(label_mapping_path)

    # Preprocessing: tokenization, lowercasing, punctuation & stop word removal, lemmatization
    def preprocess_text_spacy(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if token.text.isalnum()]
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        cleaned_text = " ".join(filtered_tokens)
        doc_cleaned = self.nlp(cleaned_text)
        lemmatized_tokens = [token.lemma_ for token in doc_cleaned]
        return " ".join(lemmatized_tokens)
    
    # Vectorize the cleaned text using TF-IDF
    def extract_feature(self, text):
        return self.vectorizer.transform([text])
    
    # Predict the category label from the input text
    def predict(self, text):
        process_text = self.preprocess_text_spacy(text)
        text_vec = self.extract_feature(process_text)
        prediction = self.model.predict(text_vec)
        return self.label_mapping.get(str(prediction[0]), "Unknown label")

# Streamlit app class
class TextCategorizationApp:
    def __init__(self):
        self.title = "Text Categorization with Machine Learning"
        self.upload_label = "Upload an English text file (.txt)"
        self.uploaded_file = None
        self.text_content = ""

    # Display the app title
    def display_title(self):
        st.markdown(
            f"<h1 style='text-align: center; color: #4B8BBE;'>{self.title}</h1>",
            unsafe_allow_html=True
        )

    # File uploader UI
    def upload_file(self):
        self.uploaded_file = st.file_uploader(
            label=self.upload_label,
            type=['txt'],
            label_visibility="visible"
        )
    
    # Read the uploaded file content
    def read_file(self):
        if self.uploaded_file is not None:
            file_bytes = self.uploaded_file.read()
            try:
                self.text_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                st.error("Decoding error: please ensure the file is in UTF-8 format.")
                self.text_content = ""

    # Main function that runs the full app logic
    def run(self):
        self.display_title()
        self.upload_file()
        self.read_file()

        if self.text_content:
            # Initialize model with paths to the model, vectorizer, and label mapping
            model = ModelCategorization(
                model_name="svm_model.joblib",
                vectorizer_path="vectorizer_tfidf.joblib",
                label_mapping_path="label_mapping.json"
            )
            # Predict category label
            prediction_label = model.predict(self.text_content)

            # Display the predicted category
            st.markdown("---")
            st.subheader("Predicted Category:")
            st.success(f"🧠 {prediction_label}")

# Run the Streamlit app
if __name__ == "__main__":
    app = TextCategorizationApp()
    app.run()
