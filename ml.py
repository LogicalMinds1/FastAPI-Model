from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
import xgboost as xgb

app = FastAPI(
    title="Pregnancy Risk Prediction API",
    description="API for predicting pregnancy risk levels based on maternal health factors",
    version="1.0.0"
)

# Load model and encoders
try:
    model = joblib.load('stats/xgb_model.pkl')
    le_y = joblib.load('stats/label_encoder_y.pkl')
    label_encoders = joblib.load('stats/label_encoders_features.pkl')
    
    # Get the exact feature order the model expects
    if isinstance(model, xgb.XGBClassifier):
        model_features = model.get_booster().feature_names
    else:
        # Fallback for non-XGBoost models or older versions
        model_features = list(label_encoders.keys()) + [
            'BP', 'BP1', 'HEMOGLOBIN', 'HEART_RATE', 'BLOOD_SUGAR', 'FEVER',
            'IFA_QUANTITY', 'NO_OF_WEEKS', 'PHQ_SCORE', 'GAD_SCORE',
            'PULSE_RATE', 'RESPIRATORY_RATE', 'UTERUS_SIZE', 'OGTT_2_HOURS',
            'KNOWN_EPILEPTIC', 'CONVULSION_SEIZURES', 'FOLIC_ACID',
            'AGE', 'HEIGHT', 'WEIGHT'
        ]
        
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

class PredictionInput(BaseModel):
    AGE: float
    HEIGHT: float
    WEIGHT: float
    BLOOD_GRP: str
    HUSBAND_BLOOD_GROUP: str
    GRAVIDA: str
    PARITY: str
    ABORTIONS: str
    PREVIOUS_ABORTION: str
    LIVE: str
    DEATH: str
    KNOWN_EPILEPTIC: float
    TWIN_PREGNANCY: str
    GESTANTIONAL_DIA: str
    CONVULSION_SEIZURES: float
    BP: float
    BP1: float
    HEMOGLOBIN: float
    PULSE_RATE: float
    RESPIRATORY_RATE: float
    HEART_RATE: float
    FEVER: float
    OEDEMA: str
    OEDEMA_TYPE: str
    UTERUS_SIZE: float
    URINE_SUGAR: str
    URINE_ALBUMIN: str
    THYROID: str
    RH_NEGATIVE: str
    SYPHYLIS: str
    HIV: str
    HIV_RESULT: str
    HEP_RESULT: str
    BLOOD_SUGAR: float
    OGTT_2_HOURS: float
    WARNING_SIGNS_SYMPTOMS_HTN: str
    ANY_COMPLAINTS_BLEEDING_OR_ABNORMAL_DISCHARGE: str
    IFA_TABLET: str
    IFA_QUANTITY: float
    IRON_SUCROSE_INJ: str
    CALCIUM: str
    FOLIC_ACID: float
    SCREENED_FOR_MENTAL_HEALTH: str
    PHQ_SCORE: float
    GAD_SCORE: float
    PHQ_ACTION: str
    GAD_ACTION: str
    ANC1FLG: str
    ANC2FLG: str
    ANC3FLG: str
    ANC4FLG: str
    MISSANC1FLG: str
    MISSANC2FLG: str
    MISSANC3FLG: str
    MISSANC4FLG: str
    NO_OF_WEEKS: float
    DELIVERY_MODE: str
    PLACE_OF_DELIVERY: str
    IS_PREV_PREG: str
    CONSANGUINITY: str

class PredictionOutput(BaseModel):
    risk_level: str
    confidence: float
    probabilities: Dict[str, float]

@app.get("/")
async def root():
    return {
        "message": "Pregnancy Risk Prediction API",
        "version": "1.0.0",
        "available_endpoints": ["/predict", "/model_info"]
    }

@app.get("/model_info")
async def get_model_info():
    return {
        "model_type": "XGBoost Classifier",
        "classes": le_y.classes_.tolist(),
        "features": model_features,
        "feature_count": len(model_features)
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Convert input to dataframe
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])
        
        # Print received columns for debugging
        print("Received columns:", df.columns.tolist())
        print("Model expects:", model_features)
        
        # Preprocess the data
        # Numeric columns
        numeric_cols = ['BP', 'BP1', 'HEMOGLOBIN', 'HEART_RATE', 'BLOOD_SUGAR', 'FEVER',
                       'IFA_QUANTITY', 'NO_OF_WEEKS', 'PHQ_SCORE', 'GAD_SCORE',
                       'PULSE_RATE', 'RESPIRATORY_RATE', 'UTERUS_SIZE', 'OGTT_2_HOURS',
                       'KNOWN_EPILEPTIC', 'CONVULSION_SEIZURES', 'FOLIC_ACID',
                       'AGE', 'HEIGHT', 'WEIGHT']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill any remaining NaNs with column mean
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
        
        # Label encode categorical features
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                unseen_mask = ~df[col].isin(le.classes_)
                if unseen_mask.any():
                    # For unseen labels, use the most frequent class
                    df.loc[unseen_mask, col] = le.classes_[0]
                df[col] = le.transform(df[col])
        
        # Ensure all expected columns are present and in correct order
        missing_cols = set(model_features) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Fill missing with 0
            
        # Reorder columns to exactly match model's expected order
        df = df[model_features]
        
        # Make prediction
        probabilities = model.predict_proba(df)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = le_y.inverse_transform([predicted_class_idx])[0]
        confidence = probabilities[predicted_class_idx]
        
        # Create probability dictionary with class labels
        prob_dict = {
            le_y.classes_[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "risk_level": predicted_class,
            "confidence": float(confidence),
            "probabilities": prob_dict
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)