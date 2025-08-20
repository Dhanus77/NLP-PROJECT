# 🧠 AI-Powered Crime Classification

This project uses Natural Language Processing (NLP) and Transformer-based models (Legal-BERT) to classify legal documents into different crime categories.

## 🔍 Features
- Preprocessing of legal text (cleaning, tokenization, stopwords)
- Feature extraction using Legal-BERT embeddings
- Classification using a neural network trained on Legal-BERT outputs
- Evaluation with precision, recall, F1-score, and confusion matrix
- Streamlit web app for interactive classification

## 📁 Project Structure

```
AI_Crime_Classification_Project/
├── app.py                  # Streamlit web app
├── crime_classifier.pth    # Trained PyTorch model
├── requirements.txt        # Python dependencies
├── notebook.ipynb          # Google Colab notebook
├── README.md               # This file
└── final_presentation.pptx # Your project PPT
```

## 🚀 Running the Web App

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

## 📌 Model Inputs
Upload a `.txt` file with legal text and the model will return the predicted crime category (e.g., Theft, Fraud, Robbery, etc.).

---

*Built using Hugging Face Transformers, PyTorch, and Streamlit.*