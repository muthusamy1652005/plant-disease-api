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

# CORS setup for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load YOLOv8 model (best.onnx file folder-la irukanum)
model = YOLO('best.onnx', task='detect')

# --- GEMINI SETUP ---
# Render Environment Variables-la GOOGLE_API_KEY set pannalaam
# Illana inga direct-aa unga key-ah paste pannunga
api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyCRc9sIAI17IxsID_EaGO_OnWiQWlcSAfU")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash", # <--- Fixed model name
    google_api_key=api_key,
    temperature=0.7
)

# --- FULL 38 CLASSES MASTER DICTIONARY ---
disease_solutions = {
    # APPLE
    'Apple___Apple_scab': {'disease': 'ஆப்பிள் செதில் நோய்', 'cause': 'வென்டுரியா பூஞ்சை', 'solution': 'கேப்டன் தெளிக்கவும்', 'prevention': 'இலைகளை அகற்றவும்'},
    'Apple___Black_rot': {'disease': 'ஆப்பிள் கருப்பு அழுகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'பாதிக்கப்பட்ட கிளைகளை கத்தரித்தல்'},
    'Apple___Cedar_apple_rust': {'disease': 'ஆப்பிள் துரு நோய்', 'cause': 'ஜிம்னோஸ்போரான்ஜியம் பூஞ்சை', 'solution': 'மைக்கோபுட்டானில் தெளிக்கவும்', 'prevention': 'ஜூனிபர் மரங்களை தவிர்க்கவும்'},
    'Apple___healthy': {'disease': 'ஆப்பிள் ஆரோக்கியமானது ✅', 'cause': 'நோய் எதுவும் இல்லை', 'solution': 'தொடர்ந்து பராமரிக்கவும்', 'prevention': 'கண்காணிப்பு அவசியம்'},
    
    # CORN / MAIZE
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {'disease': 'சோளம் சாம்பல் இலை புள்ளி', 'cause': 'செர்கோஸ்போரா பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி தெளிக்கவும்', 'prevention': 'காற்றோட்டம் ஏற்படுத்தவும்'},
    'Corn___Common_rust': {'disease': 'சோளம் துரு நோய்', 'cause': 'புக்கினியா பூஞ்சை', 'solution': 'ட்ரைஅசோல் தெளிக்கவும்', 'prevention': 'நோய் எதிர்ப்பு ரகங்கள்'},
    'Corn___Northern_Leaf_Blight': {'disease': 'சோளம் வடக்கு இலை கருகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'அசோக்ஸிஸ்ட்ரோபின் தெளிக்கவும்', 'prevention': 'பயிர் சுழற்சி'},
    'Corn___healthy': {'disease': 'சோளம் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'உரம் இடவும்', 'prevention': 'கண்காணிப்பு'},

    # POTATO
    'Potato___Early_blight': {'disease': 'உருளைக்கிழங்கு முன் கருகல்', 'cause': 'ஆல்டர்னேரியா பூஞ்சை', 'solution': 'குளோரோதலோனில்', 'prevention': 'இடைவெளி'},
    'Potato___Late_blight': {'disease': 'உருளைக்கிழங்கு பின் கருகல்', 'cause': 'பைட்டோஃப்தோரா பூஞ்சை', 'solution': 'மேன்கோசெப் தெளிக்கவும்', 'prevention': 'வடிகால் வசதி'},
    'Potato___healthy': {'disease': 'உருளைக்கிழங்கு ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # TOMATO
    'Tomato___Bacterial_spot': {'disease': 'தக்காளி பாக்டீரியா புள்ளி', 'cause': 'சாந்தோமோனாஸ் பாக்டீரியா', 'solution': 'செம்பு சல்பேட் தெளிக்கவும்', 'prevention': 'சுத்தம் பேணவும்'},
    'Tomato___Early_blight': {'disease': 'தக்காளி ஆரம்ப கருகல் நோய்', 'cause': 'ஆல்டர்னேரியா பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி தெளிக்கவும்', 'prevention': 'பயிர் சுழற்சி'},
    'Tomato___Late_blight': {'disease': 'தக்காளி தாமத கருகல் நோய்', 'cause': 'பைட்டோஃப்தோரா பூஞ்சை', 'solution': 'மேன்கோசெப் தெளிக்கவும்', 'prevention': 'வடிகால்'},
    'Tomato___Leaf_Mold': {'disease': 'தக்காளி இலை அச்சு நோய்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'காற்றோட்டம்'},
    'Tomato___Septoria_leaf_spot': {'disease': 'தக்காளி செப்டோரியா புள்ளி', 'cause': 'பூஞ்சை தொற்று', 'solution': 'பூஞ்சைக்கொல்லி தெளிக்கவும்', 'prevention': 'இலை நீக்கம்'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'disease': 'தக்காளி சிலந்தி பூச்சி', 'cause': 'பூச்சி தாக்குதல்', 'solution': 'அபாமெக்டின் தெளிக்கவும்', 'prevention': 'கண்காணிப்பு'},
    'Tomato___Target_Spot': {'disease': 'தக்காளி இலக்கு புள்ளி நோய்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'பூஞ்சைக்கொல்லி தெளிக்கவும்', 'prevention': 'இலைகளை அகற்றவும்'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'disease': 'தக்காளி மஞ்சள் வைரஸ்', 'cause': 'வெள்ளை ஈ வைரஸ்', 'solution': 'இமிடாக்லோப்ரிட் தெளிக்கவும்', 'prevention': 'வெள்ளை ஈ கட்டுப்பாடு'},
    'Tomato___Tomato_mosaic_virus': {'disease': 'தக்காளி மொசைக் வைரஸ்', 'cause': 'வைரஸ் தொற்று', 'solution': 'செடியை அகற்றவும்', 'prevention': 'சுத்தம்'},
    'Tomato___healthy': {'disease': 'தக்காளி ஆரோக்கியமான இலை ✅', 'cause': 'நோய் இல்லை', 'solution': 'பராமரிக்கவும்', 'prevention': 'கண்காணிப்பு'},
    # ... neenga unga dictionary-la irukura matha classes-um ingaye maintain pannunga ...
}

@app.get('/')
def health():
    return {'status': 'Plant Disease & AI Chat API is Live', 'classes': 38}

# --- ENDPOINT 1: YOLO PREDICTION ---
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
            
            clean_raw = raw_name.lower().replace('leaf', '').replace('leafs', '').replace('blights', 'blight').replace('rusts', 'rust').replace('(maize)', '').replace('spots', 'spot').replace(',', '').replace('(', '').replace(')', '').strip()
            clean_raw = " ".join(clean_raw.split())

            info = None
            for key, data in disease_solutions.items():
                clean_key = key.lower().replace('___', ' ').replace('_', ' ').replace('leaf', '').strip()
                clean_key = " ".join(clean_key.split())
                if clean_raw == clean_key or clean_key in clean_raw or clean_raw in clean_key:
                    info = data
                    break

            if not info:
                info = {'disease': raw_name.replace('___', ' '), 'cause': 'பூஞ்சை/வைரஸ்', 'solution': 'மருந்து தெளிக்கவும்', 'prevention': 'கண்காணிக்கவும்'}

            detections.append({
                'disease_name': info['disease'],
                'confidence': confidence,
                'cause': info['cause'],
                'solution': info['solution'],
                'prevention': info['prevention']
            })
    return {'detections': detections}

# --- ENDPOINT 2: SMART AI FARMER CHAT ---
@app.post('/chat')
async def smart_farmer_chat(user_query: str = Body(..., embed=True)):
    try:
        system_message = SystemMessage(content="""
            You are 'Zynixo Agri Bot', an expert AI Agricultural Assistant. 
            1. Help farmers diagnose plant diseases and provide solutions.
            2. Answer strictly in a mix of Tamil and English (Thanglish).
            3. Be friendly and helpful like a farming expert.
        """)
        user_message = HumanMessage(content=user_query)
        response = llm.invoke([system_message, user_message])
        return {"response": response.content}
    except Exception as e:
        return {"response": f"Sorry nanba, Gemini connect aagala: {str(e)}"}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)