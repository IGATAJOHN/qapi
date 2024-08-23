from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form,Path,Request
from pydantic import BaseModel, EmailStr, Field
from typing import Optional,List,Dict
import os
import re
from bson import ObjectId
from passlib.context import CryptContext
from passlib.hash import bcrypt
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
import openai
from openai import OpenAI
app = FastAPI()
load_dotenv()
# MongoDB setup
app.config["MONGO_URI"] = os.getenv("MONGODB_URI")
client = MongoClient(app.config["MONGO_URI"])
db = client['qapi']
# MongoDB collections
users_collection = db.users
doctors_collection = db.doctors
messages_collection = db.messages
reviews_collection = db.reviews
appointments_collection = db.appointments
favorites_collection = db.favorites
replies_collection = db.replies
user_activity_collection = db.user_activity
# Define the collection
conversations_collection = db.conversations
api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=api_key)
# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Ensure the upload directory exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads", "avatars")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Models
class Consultation(BaseModel):
    patient_id: str
    doctor_id: str
    symptoms: str
    diagnosis: Optional[str] = None
    prescription: Optional[str] = None
    completed: bool = False
    date: datetime = datetime.utcnow()

class PatientRecord(BaseModel):
    doctor_id: str
    diagnosis: str
    prescription: str
    date: datetime = datetime.utcnow()

class Review(BaseModel):
    patient_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    date: datetime = datetime.utcnow()

class Appointment(BaseModel):
    patient_id: str
    doctor_id: str
    date: datetime
    status: str = "scheduled"
def save_file(file: UploadFile, directory: str) -> str:
    filename = file.filename
    file_location = os.path.join(directory, filename)
    
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    return file_location

def validate_contact(contact: str):
    # A simple regex pattern for validating phone numbers
    pattern = re.compile(r"^\+?[1-9]\d{1,14}$")  # This pattern allows for international numbers
    if not pattern.match(contact):
        raise HTTPException(status_code=400, detail="Invalid phone number format.")
    return contact

def validate_file_format(file: UploadFile, allowed_extensions: list):
    filename = file.filename
    file_extension = filename.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File must be one of the following formats: {', '.join(allowed_extensions)}")
    return file

# Helper function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Helper function to create access token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
class UserRegistration(BaseModel):
    first_name: str = Field(..., example="John")
    last_name: str = Field(..., example="Doe")
    email: EmailStr
    password: str = Field(..., min_length=6)
    contact: str = Field(..., example="+123456789")
    role: str = Field(..., example="user")  # user or doctor
    specialization: Optional[str] = None
    location: Optional[str] = None
    about: Optional[str] = None
    experience: Optional[str] = None
    verified: bool = False
    online: bool = False
    rating: int = 0

@app.post("/register/")
async def register(
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(..., min_length=6),
    contact: str = Form(...),
    role: str = Form(..., example="user"),
    specialization: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    about: Optional[str] = Form(None),
    experience: Optional[str] = Form(None),
    doctor_avatar: Optional[UploadFile] = File(None),
    medical_license: Optional[UploadFile] = File(None),
    medical_school_certificate: Optional[UploadFile] = File(None),
    nysc_certificate: Optional[UploadFile] = File(None),
):
    # Validate the contact number
    validate_contact(contact)

    # Validate required fields based on role
    if role == "doctor" and (not specialization or not location or not about or not experience):
        raise HTTPException(status_code=400, detail="Doctor registration requires specialization, location, about, and experience.")

    # Validate file formats
    if doctor_avatar:
        validate_file_format(doctor_avatar, ["jpg", "jpeg", "png", "gif"])
    if medical_license:
        validate_file_format(medical_license, ["pdf"])
    if medical_school_certificate:
        validate_file_format(medical_school_certificate, ["pdf"])
    if nysc_certificate:
        validate_file_format(nysc_certificate, ["pdf"])

    # Hash the password
    hashed_password = bcrypt.hash(password)

    avatar_path = save_file(doctor_avatar, UPLOAD_FOLDER) if doctor_avatar else None
    medical_license_path = save_file(medical_license, UPLOAD_FOLDER) if medical_license else None
    medical_school_certificate_path = save_file(medical_school_certificate, UPLOAD_FOLDER) if medical_school_certificate else None
    nysc_certificate_path = save_file(nysc_certificate, UPLOAD_FOLDER) if nysc_certificate else None

    user_data = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "password": hashed_password,
        "contact": contact,
        "role": role,
        "avatar": avatar_path,
    }

    if role == "doctor":
        user_data.update({
            "specialization": specialization,
            "location": location,
            "about": about,
            "experience": experience,
            "verified": False,
            "rating": 0,
            "medical_license": medical_license_path,
            "medical_school_certificate": medical_school_certificate_path,
            "nysc_certificate": nysc_certificate_path,
            "online": False,  # Only for doctors
        })
        doctors_collection.insert_one(user_data)
    else:
        users_collection.insert_one(user_data)

    return {"message": "Registration successful!"}

@app.post("/login/")
async def login(
    email: EmailStr = Form(...),
    password: str = Form(...),
):
    # Check if user exists in the collection
    user = users_collection.find_one({"email": email}) or doctors_collection.find_one({"email": email})
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email or password")

    # Verify the password
    if not verify_password(password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    # Create a JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["email"]}, expires_delta=access_token_expires)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user["role"],
        "email": user["email"]
    }

# Consultations
@app.post("/consultations/")
async def initiate_consultation(consultation: Consultation):
    consultation_id = db['consultations'].insert_one(consultation.dict()).inserted_id
    return {"consultation_id": str(consultation_id)}

@app.get("/consultations/{consultation_id}")
async def get_consultation(consultation_id: str):
    consultation = db['consultations'].find_one({"_id": ObjectId(consultation_id)})
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return consultation

@app.get("/consultations/")
async def get_consultations(user_id: Optional[str] = None, doctor_id: Optional[str] = None):
    query = {}
    if user_id:
        query["patient_id"] = user_id
    if doctor_id:
        query["doctor_id"] = doctor_id
    consultations = db['consultations'].find(query)
    return list(consultations)

@app.put("/consultations/{consultation_id}/complete")
async def complete_consultation(consultation_id: str):
    result = db['consultations'].update_one(
        {"_id": ObjectId(consultation_id)},
        {"$set": {"completed": True}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return {"message": "Consultation marked as complete"}

# Patient Records
@app.post("/patients/{patient_id}/records/")
async def add_patient_record(patient_id: str, record: PatientRecord):
    record_data = record.dict()
    record_data["patient_id"] = patient_id
    record_id = db['patient_records'].insert_one(record_data).inserted_id
    return {"record_id": str(record_id)}

@app.get("/patients/{patient_id}/records/")
async def get_patient_records(patient_id: str):
    records = db['patient_records'].find({"patient_id": patient_id})
    return list(records)

@app.get("/patients/{patient_id}/records/{record_id}")
async def get_patient_record(patient_id: str, record_id: str):
    record = db['patient_records'].find_one({"_id": ObjectId(record_id), "patient_id": patient_id})
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record

@app.put("/patients/{patient_id}/records/{record_id}")
async def update_patient_record(patient_id: str, record_id: str, record: PatientRecord):
    result = db['patient_records'].update_one(
        {"_id": ObjectId(record_id), "patient_id": patient_id},
        {"$set": record.dict()}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"message": "Record updated successfully"}

# Doctor Reviews
@app.post("/doctors/{doctor_id}/reviews/")
async def add_review(doctor_id: str, review: Review):
    review_data = review.dict()
    review_data["doctor_id"] = doctor_id
    review_id = db['reviews'].insert_one(review_data).inserted_id
    return {"review_id": str(review_id)}

@app.get("/doctors/{doctor_id}/reviews/")
async def get_reviews(doctor_id: str):
    reviews = db['reviews'].find({"doctor_id": doctor_id})
    return list(reviews)

@app.get("/doctors/{doctor_id}/reviews/{review_id}")
async def get_review(doctor_id: str, review_id: str):
    review = db['reviews'].find_one({"_id": ObjectId(review_id), "doctor_id": doctor_id})
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review

# Appointments
@app.post("/appointments/")
async def schedule_appointment(appointment: Appointment):
    appointment_id = db['appointments'].insert_one(appointment.dict()).inserted_id
    return {"appointment_id": str(appointment_id)}

@app.get("/appointments/{appointment_id}")
async def get_appointment(appointment_id: str):
    appointment = db['appointments'].find_one({"_id": ObjectId(appointment_id)})
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return appointment

@app.get("/appointments/")
async def get_appointments(user_id: Optional[str] = None, doctor_id: Optional[str] = None):
    query = {}
    if user_id:
        query["patient_id"] = user_id
    if doctor_id:
        query["doctor_id"] = doctor_id
    appointments = db['appointments'].find(query)
    return list(appointments)

@app.put("/appointments/{appointment_id}/cancel")
async def cancel_appointment(appointment_id: str):
    result = db['appointments'].update_one(
        {"_id": ObjectId(appointment_id)},
        {"$set": {"status": "cancelled"}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return {"message": "Appointment cancelled successfully"}

# Dependency to get the user_id from the session or request
async def get_user_id(request: Request):
    user_id = request.session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail="User not logged in")
    return user_id

class ChatMessage(BaseModel):
    message: str

# Generate response from OpenAI
async def generate_response(conversation_history: List[Dict[str, str]]) -> str:
    try:
        completion = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are Quantum Doctor, a healthcare assistant, capable of making diagnosis based on symptoms,
                    make sure to explain diagnosis in the simplest possible way for patients to understand.
                    start by asking the patients for their name.
                    ask necessary health question about the provided medical condition to enable you make accurate diagnosis,
                    you can predict to a high degree of accuracy the potential of future occurrence of an illness in days, weeks, months, etc after a proper understanding
                    of the underlying health pattern.
                    You were trained by a team of Machine Learning Engineers led by Engineer Igata John at QuantumLabs, 
                    a division of Quantum Innovative Tech Solutions Ltd.
                    """,
                },
            ] + conversation_history
        )

        model_response = completion.choices[0].message['content'].strip()
        return model_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chatbot interaction endpoint
@app.post("/chatbot")
async def chatbot(data: ChatMessage, user_id: str = Depends(get_user_id)):
    try:
        user_input = data.message

        # Retrieve conversation history from the database
        conversation_history = await conversations_collection.find_one(
            {'user_id': user_id},
            {'_id': 0, 'history': 1}
        )
        conversation_history = conversation_history['history'] if conversation_history else []

        # Append the new user input to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Generate response
        response = await generate_response(conversation_history)

        # Append the bot response to the conversation history
        conversation_history.append({"role": "assistant", "content": response})

        # Update the conversation history in the database
        await conversations_collection.update_one(
            {'user_id': user_id},
            {'$set': {'history': conversation_history}},
            upsert=True
        )

        return JSONResponse(content={'reply': response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get chatbot response endpoint
@app.post("/get-response")
async def get_response(data: ChatMessage, user_id: str = Depends(get_user_id)):
    try:
        user_input = data.message
        if not user_input:
            raise HTTPException(status_code=400, detail="No input provided")

        # Retrieve conversation history from the database
        conversation_history = await conversations_collection.find_one(
            {'user_id': user_id},
            {'_id': 0, 'history': 1}
        )
        conversation_history = conversation_history['history'] if conversation_history else []

        # Append the new user input to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Generate response
        response_text = await generate_response(conversation_history)

        # Append the bot response to the conversation history
        conversation_history.append({"role": "assistant", "content": response_text})

        # Update the conversation history in the database
        await conversations_collection.update_one(
            {'user_id': user_id},
            {'$set': {'history': conversation_history}},
            upsert=True
        )

        # Log user activity (optional, not implemented here)
        # log_user_activity(user_id, "chat_interaction", {"user_input": user_input, "response": response_text})

        return JSONResponse(content={'response': response_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get conversation history endpoint
@app.get("/get-conversation-history")
async def get_conversation_history(user_id: str = Depends(get_user_id)):
    try:
        # Retrieve conversation history from the database
        conversation_history = await conversations_collection.find_one(
            {'user_id': user_id},
            {'_id': 0, 'history': 1}
        )
        conversation_history = conversation_history['history'] if conversation_history else []

        return JSONResponse(content={'conversation_history': conversation_history})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))