from fastapi import FastAPI, File, UploadFile
import pandas as pd
import nest_asyncio
import uvicorn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse

# Initialize the FastAPI app with the correct root_path
app = FastAPI(root_path="/fastapi-app")

# Add middleware to handle CORS (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for your use case
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to accept requests from any host
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}

def init():
    global model, tokenizer
    # Load the model from a local path
    model_path = "fine_tuned_mBert"  # Ensure this path is correct and contains all necessary files
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the Excel file
    df = pd.read_excel(file.file)
    
    # Ensure the column name is 'Message'
    if 'Message' not in df.columns:
        return {"error": "Excel file must contain a 'Message' column"}

    # Drop rows with NaN values in 'Message' column or replace them with a placeholder
    df['Message'] = df['Message'].fillna("")

    # Ensure all special characters, spaces, and links are preserved
    texts = df['Message'].astype(str).tolist()

    # Tokenize the texts, ensuring special characters, links, and spaces are retained
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Predict with the model
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).tolist()
        predicted_labels = torch.argmax(logits, dim=-1).tolist()

    # Map labels to text
    classes = model.config.id2label
    label_texts = ["Inappropriate" if label == 1 else "Appropriate" for label in predicted_labels]

    # Add results to DataFrame
    df['Label'] = label_texts
    df['Confidence'] = [prob[label] for prob, label in zip(probabilities, predicted_labels)]

    # Save the results to a new Excel file
    output_file = "output.xlsx"
    df.to_excel(output_file, index=False)

    # Return the file as a response
    return FileResponse(output_file, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=output_file)

# Initialize the model and tokenizer
init()

# Required to run uvicorn in notebooks
nest_asyncio.apply()

# Run the app with uvicorn, enabling proxy headers
uvicorn.run(
    app,
    host="0.0.0.0",
    port=80,
    proxy_headers=True,
    log_level="info"
)
