import sys
import os
sys.path.insert(0, os.getcwd())

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from datetime import datetime
import hashlib
import hmac

try:
    from backend.preprocessing import preprocess_image
    from backend.model import FractureDetectionModel, model_instance
    from backend.database_utils import save_prediction, get_history, init_db, create_user, get_user_hash
except ImportError as e:
    print(f"LOCAL IMPORT ERROR: {e}")
    # Try absolute if relative failed
    from preprocessing import preprocess_image
    from model import FractureDetectionModel, model_instance
    from database_utils import save_prediction, get_history, init_db, create_user, get_user_hash

import sqlite3
try:
    from fpdf import FPDF
except ImportError as e:
    print(f"FPDF IMPORT ERROR: {e}")
    import fpdf
    from fpdf import FPDF

import base64

# Simple hash function for passwords (using hashlib instead of bcrypt due to compatibility issues)
def hash_password(password: str) -> str:
    """Hash password using SHA256 with salt"""
    salt = "osteo_scan_ai_salt"  # In production, use a real salt
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

app = FastAPI(title="Bone Fracture Detection API")

# Models for Request/Response
class User(BaseModel):
    username: str
    password: str

class ReportRequest(BaseModel):
    username: str
    prediction: str
    confidence: float
    location: str
    severity: str
    specialist: str
    image_name: str

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth Endpoints
@app.post("/register")
def register(user: User):
    try:
        hashed_password = hash_password(user.password)
        if create_user(user.username, hashed_password):
            return {"message": "User registered successfully"}
        else:
            raise HTTPException(status_code=400, detail="Username already exists")
    except Exception as e:
        print(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/login")
def login(user: User):
    try:
        stored_hash = get_user_hash(user.username)
        if not stored_hash or not verify_password(user.password, stored_hash):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        return {"message": "Login successful", "username": user.username}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/generate_report")
def generate_report(req: ReportRequest):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="OsteoScan AI - Bone Fracture Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient ID: {req.username}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Diagnosis Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Result: {req.prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {req.confidence}%", ln=True)
    
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Location: {req.location}", ln=True)
    pdf.cell(200, 10, txt=f"Severity: {req.severity}", ln=True)
    pdf.cell(200, 10, txt=f"Consult: {req.specialist}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="Disclaimer: This AI-generated report is for research and demonstration. Consult a doctor for medical advice.")
    
    # Save temporarily
    if not os.path.exists('uploads/reports'): os.makedirs('uploads/reports')
    report_path = f"uploads/reports/report_{req.username}.pdf"
    pdf.output(report_path)
    
    with open(report_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode('utf-8')
        
    return {"pdf_content": pdf_b64, "filename": f"Report_{req.username}.pdf"}

@app.on_event("startup")
def startup():
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    init_db()

@app.get("/")
def home():
    return {"message": "Bone Fracture Detection System is running"}

@app.post("/predict")
async def predict(username: str, model_name: str = "efficientnet", file: UploadFile = File(...)):
    try:
        # 1. Save uploaded image
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Preprocess image
        processed_img = preprocess_image(file_path)
        
        # 3. Predict
        label, confidence, details, heatmap_b64 = model_instance.predict(processed_img, model_name)
        
        # 4. Save to database history
        try:
            save_prediction(username, file.filename, label, confidence)
        except Exception as db_err:
            print(f"Database error: {db_err}")
        
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "image": file.filename,
            "details": details,
            "heatmap": heatmap_b64
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"SERVER ERROR: {error_trace}")
        return {
            "error": str(e),
            "trace": error_trace,
            "prediction": "Error",
            "confidence": 0
        }

@app.get("/history/{username}")
def history(username: str):
    rows = get_history(username)
    return [{"image": r[0], "prediction": r[1], "confidence": r[2], "timestamp": r[3]} for r in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
