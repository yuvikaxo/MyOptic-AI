from fastapi import FastAPI, HTTPException, File, UploadFile
import io
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import requests
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Load CNN Model ------------------
class MyopiaClassifier(nn.Module):
    def __init__(self):
        super(MyopiaClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Single neuron output for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x  # Raw logits (before activation)

# Load trained model
MODEL_PATH = "myopia_classifier.pth"
model = MyopiaClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ------------------ CNN Prediction Endpoint ------------------
@app.post("/predict_cnn")
async def predict_cnn(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            output = model(image)
            probability = torch.sigmoid(output).item()  # Convert logits to probability

        # Determine prediction
        prediction = "Myopic" if probability >= 0.5 else "Not Myopic"

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}

# Load and Train RFC Model
df = pd.read_csv("myopia.csv", sep=';')
train, test = np.split(df.sample(frac=1, random_state=42), [int(0.8 * len(df))])
x_train = train.drop(columns=["MYOPIC", "ID", "STUDYYEAR"])
y_train = train["MYOPIC"].values
x_test = test.drop(columns=["MYOPIC", "ID", "STUDYYEAR"])
y_test = test["MYOPIC"].values

categorical_cols = ["GENDER"]
numerical_cols = [col for col in x_train.columns if col not in categorical_cols]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
])
model_rfc = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])
model_rfc.fit(x_train, y_train)

# RFC Input Model
class RFCInput(BaseModel):
    GENDER: str  # Example: "Male" or "Female"
    AL: float  # Axial Length
    ACD: float  # Anterior Chamber Depth
    LT: float  # Lens Thickness
    VCD: float  # Vitreous Chamber Depth
    SPHEQ: float  # Spherical Equivalent Refraction
    
@app.post("/predict_rfc")
async def predict_rfc(data: RFCInput):
    try:
        # Convert user input into a DataFrame
        user_input_df = pd.DataFrame([{**data.dict()}])
        
        # Ensure column order matches training data
        user_input_df = user_input_df.reindex(columns=x_train.columns, fill_value=0)
        
        # Predict probabilities
        probabilities = model_rfc.predict_proba(user_input_df)[0]
        myopia_prob = probabilities[1] * 100
        not_myopia_prob = probabilities[0] * 100
        
        # Determine final prediction
        prediction = "Myopic" if myopia_prob >= 50 else "Not Myopic"
        
        return {
            "prediction": prediction,
            "confidence": max(myopia_prob, not_myopia_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chatbot Endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
MYOPTIC_CONTEXT = """
MyOptic AI is an AI-driven healthcare assistant focused on eye health.
It helps users by analyzing retinal scans, tracking eyesight changes,
and predicting risks of myopia, glaucoma, and retinal detachment.
It provides users with insights on eye care, lens usage, and vision correction.
It's owned and run by Yuvika Gupta, Mahi Tyagi, Rishika Singh, and Manan Singh.
"""

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat(data: ChatInput):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "prompt": f"{MYOPTIC_CONTEXT}\nUser: {data.message}\nAI:",
            "stream": False,
            "options": {
                "num_predict": 30,
                "temperature": 0.2
            }
        })
        return {"reply": response.json().get("response", "Error in response")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "FastAPI Server is Running!"}