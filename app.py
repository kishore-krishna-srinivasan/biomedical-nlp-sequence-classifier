from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import uvicorn
import logging

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Load the model
model = tf.keras.models.load_model("ffnn_tfidf_model.h5")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load the label encoder
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    try:
        # Tokenize the input text
        tokens = input.text.split()
        
        # Transform the input text using the TF-IDF vectorizer
        input_vector = vectorizer.transform(tokens).toarray()

        # Ensure the input vector shape matches the model's expected input shape
        if input_vector.shape[1] != model.input_shape[1]:
            raise ValueError(f"Input shape mismatch: expected {model.input_shape[1]}, got {input_vector.shape[1]}")

        # Make predictions using the model
        predictions = model.predict(input_vector)
        predicted_classes = np.argmax(predictions, axis=1)

        # Decode the predicted class to original labels
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        # Log the input and prediction
        logging.info(f"Input: {input.text}, Prediction: {predicted_labels.tolist()}")

        return {"predictions": predicted_labels.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
