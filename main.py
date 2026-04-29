from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os

# --- LANGCHAIN & GEMINI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage 

app = FastAPI(title='Plant Disease Detection & Smart AI Farmer')

# CORS setup for Flutter (Public Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load YOLOv8 model
model = YOLO('best.onnx', task='detect')

# --- GEMINI SETUP (SECURE) ---
api_key = os.getenv("GOOGLE_API_KEY")

# Fallback only for testing
if not api_key:
    api_key = "AIzaSyDvzFrw8u0svtyrVPeO5Ck1vao8kryjJe4"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=api_key,
    temperature=0.7
)

# --- FULL 38 CLASSES MASTER DICTIONARY ---
disease_solutions = {
    # APPLE
    'Apple___Apple_scab': {'disease': 'ஆப்பிள் செதில் நோய்', 'cause': 'வென்டுரியா பூஞ்சை', 'solution': 'கேப்டன் தெளிக்கவும்', 'prevention': 'இலைகளை அகற்றவும்'},
    'Apple___Black_rot': {'disease': 'ஆப்பிள் கருப்பு அழுகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'கிளைகளை கத்தரித்தல்'},
    'Apple___Cedar_apple_rust': {'disease': 'ஆப்பிள் துரு நோய்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'மைக்கோபுட்டானில் தெளிக்கவும்', 'prevention': 'சுற்றுப்புற சுத்தம்'},
    'Apple___healthy': {'disease': 'ஆப்பிள் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'தொடர்ந்து பராமரிக்கவும்', 'prevention': 'கண்காணிப்பு'},

    # BLUEBERRY
    'Blueberry___healthy': {'disease': 'புளுபெர்ரி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # CHERRY
    'Cherry_(including_sour)___Powdery_mildew': {'disease': 'செர்ரி சாம்பல் நோய்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'கந்தகம் தெளிக்கவும்', 'prevention': 'காற்றோட்டம்'},
    'Cherry_(including_sour)___healthy': {'disease': 'செர்ரி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # CORN / MAIZE
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'disease': 'சோளம் சாம்பல் இலை புள்ளி', 'cause': 'செர்கோஸ்போரா பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'பயிர் சுழற்சி'},
    'Corn_(maize)___Common_rust': {'disease': 'சோளம் துரு நோய்', 'cause': 'புக்கினியா பூஞ்சை', 'solution': 'ட்ரைஅசோல் தெளிக்கவும்', 'prevention': 'நோய் எதிர்ப்பு ரகங்கள்'},
    'Corn_(maize)___Northern_Leaf_Blight': {'disease': 'சோளம் வடக்கு இலை கருகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'அசோக்ஸிஸ்ட்ரோபின்', 'prevention': 'இலை நீக்கம்'},
    'Corn_(maize)___healthy': {'disease': 'சோளம் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'உரம் இடவும்', 'prevention': 'கண்காணிப்பு'},

    # GRAPE
    'Grape___Black_rot': {'disease': 'திராட்சை கருப்பு அழுகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'கத்தரித்தல்'},
    'Grape___Esca_(Black_Measles)': {'disease': 'திராட்சை எஸ்கா நோய்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'பாதிக்கப்பட்ட பகுதியை நீக்கவும்', 'prevention': 'முறையான நீர் மேலாண்மை'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'disease': 'திராட்சை இலை கருகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'மருந்து தெளிக்கவும்', 'prevention': 'சுத்தம்'},
    'Grape___healthy': {'disease': 'திராட்சை ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # ORANGE
    'Orange___Haunglongbing_(Citrus_greening)': {'disease': 'ஆரஞ்சு பச்சை நோய்', 'cause': 'பாக்டீரியா தொற்று', 'solution': 'செடியை அகற்றவும்', 'prevention': 'பூச்சி கட்டுப்பாடு'},

    # PEACH
    'Peach___Bacterial_spot': {'disease': 'பீச் பாக்டீரியா புள்ளி', 'cause': 'பாக்டீரியா தொற்று', 'solution': 'செம்பு தெளிக்கவும்', 'prevention': 'இலை உதிர்வை தடுத்தல்'},
    'Peach___healthy': {'disease': 'பீச் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # PEPPER BELL
    'Pepper,_bell___Bacterial_spot': {'disease': 'மிளகாய் பாக்டீரியா புள்ளி', 'cause': 'பாக்டீரியா தொற்று', 'solution': 'செம்பு சல்பேட்', 'prevention': 'சுத்தம்'},
    'Pepper,_bell___healthy': {'disease': 'மிளகாய் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # POTATO
    'Potato___Early_blight': {'disease': 'உருளைக்கிழங்கு முன் கருகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'இடைவெளி'},
    'Potato___Late_blight': {'disease': 'உருளைக்கிழங்கு பின் கருகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'மேன்கோசெப் தெளிக்கவும்', 'prevention': 'வடிகால்'},
    'Potato___healthy': {'disease': 'உருளைக்கிழங்கு ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # RASPBERRY
    'Raspberry___healthy': {'disease': 'ராஸ்பெர்ரி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # SOYBEAN
    'Soybean___healthy': {'disease': 'சோயாபீன் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # SQUASH
    'Squash___Powdery_mildew': {'disease': 'பூசணி சாம்பல் நோய்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'காற்றோட்டம்'},

    # STRAWBERRY
    'Strawberry___Leaf_scorch': {'disease': 'ஸ்ட்ராபெர்ரி இலை கருகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'மருந்து தெளிக்கவும்', 'prevention': 'தண்ணீர் தேங்காமல் பார்த்தல்'},
    'Strawberry___healthy': {'disease': 'ஸ்ட்ராபெர்ரி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # TOMATO
    'Tomato___Bacterial_spot': {'disease': 'தக்காளி பாக்டீரியா புள்ளி', 'cause': 'பாக்டீரியா', 'solution': 'செம்பு தெளிக்கவும்', 'prevention': 'சுத்தம்'},
    'Tomato___Early_blight': {'disease': 'தக்காளி ஆரம்ப கருகல்', 'cause': 'பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'பயிர் சுழற்சி'},
    'Tomato___Late_blight': {'disease': 'தக்காளி தாமத கருகல்', 'cause': 'பூஞ்சை', 'solution': 'மேன்கோசெப்', 'prevention': 'வடிகால்'},
    'Tomato___Leaf_Mold': {'disease': 'தக்காளி இலை அச்சு', 'cause': 'பூஞ்சை', 'solution': 'மருந்து தெளிக்கவும்', 'prevention': 'காற்றோட்டம்'},
    'Tomato___Septoria_leaf_spot': {'disease': 'தக்காளி செப்டோரியா புள்ளி', 'cause': 'பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'இலை நீக்கம்'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'disease': 'தக்காளி சிலந்தி பூச்சி', 'cause': 'பூச்சி', 'solution': 'அபாமெக்டின்', 'prevention': 'கண்காணிப்பு'},
    'Tomato___Target_Spot': {'disease': 'தக்காளி இலக்கு புள்ளி', 'cause': 'பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'சுத்தம்'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'disease': 'தக்காளி மஞ்சள் வைரஸ்', 'cause': 'வைரஸ்', 'solution': 'இமிடாக்லோப்ரிட்', 'prevention': 'வெள்ளை ஈ கட்டுப்பாடு'},
    'Tomato___Tomato_mosaic_virus': {'disease': 'தக்காளி மொசைக் வைரஸ்', 'cause': 'வைரஸ்', 'solution': 'செடியை நீக்கவும்', 'prevention': 'சுத்தம்'},
    'Tomato___healthy': {'disease': 'தக்காளி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'}
}

@app.get('/')
def health():
    return {'status': 'Plant Disease API Live', 'classes': 38}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    results = model(img)
    detections = []

    for r in results:
        for box in r.boxes:
            raw_name = model.names[int(box.cls)] 
            confidence = round(float(box.conf) * 100, 2)
            
            info = disease_solutions.get(raw_name)
            if not info:
                # Fuzzy matching search if exact key is missed
                for key, data in disease_solutions.items():
                    if raw_name.lower() in key.lower():
                        info = data
                        break
            
            if not info:
                info = {'disease': raw_name, 'cause': 'தெரியவில்லை', 'solution': 'நிபுணரை அணுகவும்', 'prevention': 'கண்காணிப்பு'}

            detections.append({
                'disease_name': info['disease'],
                'confidence': confidence,
                'cause': info['cause'],
                'solution': info['solution'],
                'prevention': info['prevention']
            })
    return {'detections': detections}

@app.post('/chat')
async def smart_farmer_chat(user_query: str = Body(..., embed=True)):
    try:
        system_message = SystemMessage(content="You are expert AI Agricultural Assistant 'Zynixo Agri Bot'. Help farmers in Tamil and English.")
        user_message = HumanMessage(content=user_query)
        response = llm.invoke([system_message, user_message])
        return {"response": response.content}
    except Exception as e:
        return {"response": "Sorry nanba, Gemini connect aagala. Madiyum try pannunga."}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host='0.0.0.0', port=port)