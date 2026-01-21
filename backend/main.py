from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import sys
import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import fused model wrapper
from model_wrapper import FusedModel

app = FastAPI(title="OvaScan AI Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, restrict to frontend domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Firebase Initialization ---
db = None
try:
    cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serviceAccountKey.json')
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized successfully")
    else:
        print(f"Warning: {cred_path} not found. Firebase features will be disabled.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")

async def save_prediction_to_db(data):
    """Saves prediction result to Firestore"""
    if db:
        try:
            # Add timestamp
            data['timestamp'] = datetime.datetime.now()
            # Add to 'predictions' collection
            db.collection('predictions').add(data)
            print("Prediction saved to database")
        except Exception as e:
            print(f"Failed to save to database: {e}")
    else:
        print("Database not initialized, skipping save")

# --- Model Loading ---
# Load fused model
base_dir = os.path.dirname(os.path.abspath(__file__))
fused_model = None

try:
    print(f"Loading fused model from fused_model.pkl...")
    fused_model = FusedModel('fused_model.pkl')
    print("Fused model loaded successfully")
except Exception as e:
    print(f"Error loading fused model: {e}")
    import traceback
    traceback.print_exc()
    # Don't crash, just log error

# Clinical findings and recommendations mapping
CLINICAL_INFO = {
    "class0_notinfected": {
        "findings": "No pathological abnormalities detected. Ovarian tissue structure appears normal with no signs of inflammation or tumor growth.",
        "recommendations": [
            "Routine annual gynecological screening",
            "Maintain healthy lifestyle",
            "Report any new symptoms immediately"
        ]
    },
    "class1_infected": {
        "findings": "Signs of inflammation detected consistent with infection. Tissue texture indicates inflammatory response.",
        "recommendations": [
            "Consult for antibiotic therapy",
            "Follow-up imaging in 4-6 weeks",
            "Screen for pelvic inflammatory disease (PID)"
        ]
    },
    "class2_ovariancancer": {
        "findings": "Malignant characteristics observed. Irregular borders and heterogeneous echogenicity suggestive of carcinoma.",
        "recommendations": [
            "Immediate oncological referral",
            "Further staging (CT/MRI)",
            "Biopsy for histopathological confirmation",
            "CA-125 blood test"
        ]
    },
    "class3_ovariantumor": {
        "findings": "Benign tumor characteristics observed. Well-defined margins and homogeneous echogenicity suggestive of benign neoplasm.",
        "recommendations": [
            "Schedule surgical consultation",
            "Regular monitoring (ultrasound every 3-6 months)",
            "Consider tumor markers to rule out malignancy"
        ]
    }
}

@app.get("/")
async def root():
    status = "running"
    db_status = "connected" if db else "not_configured"
    return {"message": "OvaScan AI Backend is running", "status": status, "database": db_status}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if fused_model is None:
        raise HTTPException(status_code=503, detail="Fused model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Use fused model to predict directly
        result = fused_model.predict(image)
        
        # Format response
        predicted_class = result['predicted_class']
        confidence = float(result['confidence']) # Ensure native python float for JSON/Firestore serialization
        
        info = CLINICAL_INFO.get(predicted_class, {
            "findings": "Classification successful.",
            "recommendations": ["Consult a specialist"]
        })
        
        # Map class name to display name
        display_name_map = {
            "class0_notinfected": "Not Infected (Healthy)",
            "class1_infected": "Infected (Inflammatory)",
            "class2_ovariancancer": "Ovarian Cancer (Malignant)",
            "class3_ovariantumor": "Ovarian Tumor (Benign)"
        }
        
        diagnosis_display = display_name_map.get(predicted_class, predicted_class)

        response_data = {
            "Diagnosis": diagnosis_display,
            "Confidence Score": confidence,
            "Clinical Findings": info["findings"],
            "Recommendations": info["recommendations"]
        }
        
        # Save to database (asynchronously friendly)
        db_data = {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "diagnosis": diagnosis_display,
            "confidence": confidence,
            "findings": info["findings"]
        }
        await save_prediction_to_db(db_data)

        return JSONResponse(content=response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
