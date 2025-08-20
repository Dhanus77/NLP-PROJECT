import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import numpy as np

# Load tokenizer and Legal-BERT model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Define the classifier network
class ClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Load the trained model
input_dim = 768
num_classes = 8
model = ClassifierNN(input_dim, num_classes)
model.load_state_dict(torch.load("crime_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Rebuild label encoder
le = LabelEncoder()
le.classes_ = np.array([
    "Arson", "Assault", "Burglary", "Fraud", 
    "Murder", "Robbery", "Theft", "Vandalism"
])


# Streamlit UI
st.title("🚨 AI-Powered Crime Classification")
st.markdown("Upload a legal text file to predict the crime category.")

uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
if uploaded_file:
    try:
        # Attempt UTF-8 decoding first
        text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        # Fallback if decoding error occurs
        text = uploaded_file.read().decode("ISO-8859-1")

    st.subheader("Document Preview:")
    st.write(text[:500] + "...")

    # Tokenize input
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        bert_output = bert_model(**encoded)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
        logits = model(cls_embedding)
        pred_idx = logits.argmax(dim=1).item()
        pred_label = le.inverse_transform(np.array([pred_idx]))[0]

    st.success(f"**Predicted Crime Category:** {pred_label}")
