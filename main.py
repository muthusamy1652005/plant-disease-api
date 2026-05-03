from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

app = FastAPI(
    title='Plant Leaf Disease Detection API',
    description='YOLOv8 + Gemini AI - Tamil Language',
    version='3.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load YOLOv8 model
model = YOLO('best.onnx', task='detect')

# Gemini Setup
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = "AIzaSyCdnRPf6HS6Q08ujMrmO_F0qoJC9isQ9F0"  # Replace with your actual API key or set as environment variable

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

# Full 38 Classes Tamil Disease Solutions
disease_solutions = {
    # APPLE
    'Apple___Apple_scab': {
        'disease': 'ஆப்பிள் செதில் நோய்',
        'cause': 'வென்டுரியா இனாக்வாலிஸ் பூஞ்சை காரணமாக வருகிறது',
        'solution': 'கேப்டன் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'விழுந்த இலைகளை அகற்றி அழிக்கவும்'
    },
    'Apple___Black_rot': {
        'disease': 'ஆப்பிள் கருப்பு அழுகல் நோய்',
        'cause': 'போட்ரியோஸ்பேரியா ஒப்டியுசா பூஞ்சை காரணமாக வருகிறது',
        'solution': 'மைக்லோபுட்டானில் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'பாதிக்கப்பட்ட கிளைகளை கத்தரிக்கவும்'
    },
    'Apple___Cedar_apple_rust': {
        'disease': 'ஆப்பிள் கேதார் துரு நோய்',
        'cause': 'ஜிம்னோஸ்போரான்ஜியம் காரணமாக வருகிறது',
        'solution': 'மைக்கோபுட்டானில் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'அருகில் ஜூனிபர் மரங்கள் வளர்க்காதீர்கள்'
    },
    'Apple___healthy': {
        'disease': 'ஆப்பிள் ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # BLUEBERRY
    'Blueberry___healthy': {
        'disease': 'ப்ளூபெரி ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # CHERRY
    'Cherry_(including_sour)___Powdery_mildew': {
        'disease': 'செர்ரி பவுடரி மில்டியூ நோய்',
        'cause': 'போடோஸ்பேரா சேராசி பூஞ்சை காரணமாக வருகிறது',
        'solution': 'கந்தக அடிப்படையிலான பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'அதிக ஈரப்பதம் தவிர்க்கவும்'
    },
    'Cherry_(including_sour)___healthy': {
        'disease': 'செர்ரி ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # CORN
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'disease': 'சோளம் சாம்பல் இலை புள்ளி நோய்',
        'cause': 'செர்கோஸ்போரா சீடினா பூஞ்சை காரணமாக வருகிறது',
        'solution': 'ஸ்ட்ரோபிலுரின் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'நல்ல காற்றோட்டம் ஏற்படுத்தவும்'
    },
    'Corn_(maize)___Common_rust': {
        'disease': 'சோளம் பொதுவான துரு நோய்',
        'cause': 'புக்கினியா சோர்கி பூஞ்சை காரணமாக வருகிறது',
        'solution': 'ட்ரைஅசோல் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'நோய் எதிர்ப்பு சக்தி உள்ள ரகங்கள் பயன்படுத்தவும்'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'disease': 'சோளம் வடக்கு இலை கருகல் நோய்',
        'cause': 'எக்ஸெரோஹிலம் துர்சிகம் பூஞ்சை காரணமாக வருகிறது',
        'solution': 'அசோக்ஸிஸ்ட்ரோபின் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'பயிர் சுழற்சி செய்யவும்'
    },
    'Corn_(maize)___healthy': {
        'disease': 'சோளம் ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # Old Corn keys (backward compatibility)
    'Corn___Common_rust': {
        'disease': 'சோளம் பொதுவான துரு நோய்',
        'cause': 'புக்கினியா சோர்கி பூஞ்சை காரணமாக வருகிறது',
        'solution': 'ட்ரைஅசோல் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'நோய் எதிர்ப்பு சக்தி உள்ள ரகங்கள் பயன்படுத்தவும்'
    },
    'Corn___Northern_Leaf_Blight': {
        'disease': 'சோளம் வடக்கு இலை கருகல் நோய்',
        'cause': 'எக்ஸெரோஹிலம் துர்சிகம் பூஞ்சை காரணமாக வருகிறது',
        'solution': 'அசோக்ஸிஸ்ட்ரோபின் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'பயிர் சுழற்சி செய்யவும்'
    },
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
        'disease': 'சோளம் சாம்பல் இலை புள்ளி நோய்',
        'cause': 'செர்கோஸ்போரா சீடினா பூஞ்சை காரணமாக வருகிறது',
        'solution': 'ஸ்ட்ரோபிலுரின் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'நல்ல காற்றோட்டம் ஏற்படுத்தவும்'
    },
    'Corn___healthy': {
        'disease': 'சோளம் ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # GRAPE
    'Grape___Black_rot': {
        'disease': 'திராட்சை கருப்பு அழுகல் நோய்',
        'cause': 'குயிக்னார்டியா பிட்வெல்லி பூஞ்சை காரணமாக வருகிறது',
        'solution': 'மேன்கோசெப் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'பாதிக்கப்பட்ட பழங்களை அகற்றவும்'
    },
    'Grape___Esca_(Black_Measles)': {
        'disease': 'திராட்சை எஸ்கா நோய்',
        'cause': 'பாஸ்கிரியோஸ்போரா பூஞ்சை காரணமாக வருகிறது',
        'solution': 'பாதிக்கப்பட்ட கொடிகளை வெட்டி அகற்றவும்',
        'prevention': 'கத்தரிக்கும் கருவிகளை கிருமிநாசினி செய்யவும்'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'disease': 'திராட்சை இலை கருகல் நோய்',
        'cause': 'இசரியோப்சிஸ் பூஞ்சை காரணமாக வருகிறது',
        'solution': 'செம்பு அடிப்படையிலான பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'தோட்டத்தில் காற்றோட்டம் ஏற்படுத்தவும்'
    },
    'Grape___healthy': {
        'disease': 'திராட்சை ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # ORANGE
    'Orange___Haunglongbing_(Citrus_greening)': {
        'disease': 'ஆரஞ்சு சிட்ரஸ் பச்சையாகும் நோய்',
        'cause': 'கேன்டிடேட்டஸ் லிபரிபேக்டர் பாக்டீரியா காரணமாக வருகிறது',
        'solution': 'டெட்ராசைக்ளின் ஆன்டிபயாடிக் செலுத்தவும்',
        'prevention': 'சைலிட் பூச்சிகளை கட்டுப்படுத்தவும்'
    },

    # PEACH
    'Peach___Bacterial_spot': {
        'disease': 'பீச் பாக்டீரியா புள்ளி நோய்',
        'cause': 'சாந்தோமோனாஸ் பாக்டீரியா காரணமாக வருகிறது',
        'solution': 'செம்பு ஹைட்ராக்சைடு தெளிக்கவும்',
        'prevention': 'நோயற்ற மரக்கன்றுகள் பயன்படுத்தவும்'
    },
    'Peach___healthy': {
        'disease': 'பீச் ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # PEPPER
    'Pepper,_bell___Bacterial_spot': {
        'disease': 'மிளகாய் பாக்டீரியா புள்ளி நோய்',
        'cause': 'சாந்தோமோனாஸ் காம்பெஸ்ட்ரிஸ் பாக்டீரியா காரணமாக வருகிறது',
        'solution': 'செம்பு அடிப்படையிலான மருந்து தெளிக்கவும்',
        'prevention': 'நோயற்ற விதைகள் பயன்படுத்தவும்'
    },
    'Pepper,_bell___healthy': {
        'disease': 'மிளகாய் ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # POTATO
    'Potato___Early_blight': {
        'disease': 'உருளைக்கிழங்கு ஆரம்ப கருகல் நோய்',
        'cause': 'ஆல்டர்னேரியா சோலானி பூஞ்சை காரணமாக வருகிறது',
        'solution': 'குளோரோதலோனில் பூஞ்சைக்கொல்லி பயன்படுத்தவும்',
        'prevention': 'சரியான தாவர இடைவெளி பராமரிக்கவும்'
    },
    'Potato___Late_blight': {
        'disease': 'உருளைக்கிழங்கு தாமத கருகல் நோய்',
        'cause': 'பைட்டோஃப்தோரா இன்ஃபெஸ்டன்ஸ் காரணமாக வருகிறது',
        'solution': 'மேட்டலாக்சில் கலந்த பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'குளிர்ச்சியான காலநிலையில் கண்காணிக்கவும்'
    },
    'Potato___healthy': {
        'disease': 'உருளைக்கிழங்கு ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # RASPBERRY
    'Raspberry___healthy': {
        'disease': 'ராஸ்பெரி ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # SOYBEAN
    'Soybean___healthy': {
        'disease': 'சோயாபீன் ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # SQUASH
    'Squash___Powdery_mildew': {
        'disease': 'ஸ்குவாஷ் பவுடரி மில்டியூ நோய்',
        'cause': 'போடோஸ்பேரா சாந்தி பூஞ்சை காரணமாக வருகிறது',
        'solution': 'கந்தக பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'காற்றோட்டமான சூழ்நிலை பராமரிக்கவும்'
    },

    # STRAWBERRY
    'Strawberry___Leaf_scorch': {
        'disease': 'ஸ்ட்ராபெரி இலை கருகல் நோய்',
        'cause': 'டைடோஸ்பேரெல்லா பூஞ்சை காரணமாக வருகிறது',
        'solution': 'கேப்டன் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'பாதிக்கப்பட்ட இலைகளை அகற்றவும்'
    },
    'Strawberry___healthy': {
        'disease': 'ஸ்ட்ராபெரி ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },

    # TOMATO
    'Tomato___Bacterial_spot': {
        'disease': 'தக்காளி பாக்டீரியா புள்ளி நோய்',
        'cause': 'சாந்தோமோனாஸ் பாக்டீரியா காரணமாக வருகிறது',
        'solution': 'செம்பு சல்பேட் கரைசல் தெளிக்கவும்',
        'prevention': 'நோயற்ற விதைகள் பயன்படுத்தவும்'
    },
    'Tomato___Early_blight': {
        'disease': 'தக்காளி ஆரம்ப கருகல் நோய்',
        'cause': 'ஆல்டர்னேரியா சோலானி பூஞ்சை காரணமாக வருகிறது',
        'solution': 'செம்பு அடிப்படையிலான பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'பயிர் சுழற்சி கடைப்பிடிக்கவும்'
    },
    'Tomato___Late_blight': {
        'disease': 'தக்காளி தாமத கருகல் நோய்',
        'cause': 'பைட்டோஃப்தோரா இன்ஃபெஸ்டன்ஸ் பூஞ்சை காரணமாக வருகிறது',
        'solution': 'மேன்கோசெப் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'நீர் தேங்காமல் வடிகால் ஏற்படுத்தவும்'
    },
    'Tomato___Leaf_Mold': {
        'disease': 'தக்காளி இலை அச்சு நோய்',
        'cause': 'க்ளாடோஸ்போரியம் பூஞ்சை காரணமாக வருகிறது',
        'solution': 'குளோரோதலோனில் பூஞ்சைக்கொல்லி பயன்படுத்தவும்',
        'prevention': 'காற்றோட்டமான சூழ்நிலை பராமரிக்கவும்'
    },
    'Tomato___Septoria_leaf_spot': {
        'disease': 'தக்காளி செப்டோரியா இலை புள்ளி நோய்',
        'cause': 'செப்டோரியா லைக்கோபெர்சிசி பூஞ்சை காரணமாக வருகிறது',
        'solution': 'குளோரோதலோனில் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'கீழ் இலைகளை நீக்கி காற்றோட்டம் ஏற்படுத்தவும்'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'disease': 'தக்காளி சிலந்தி பூச்சி நோய்',
        'cause': 'டெட்ரானைக்கஸ் உர்டிகே பூச்சி காரணமாக வருகிறது',
        'solution': 'அபாமெக்டின் பூச்சிக்கொல்லி தெளிக்கவும்',
        'prevention': 'தாவரங்களை தொடர்ந்து கண்காணிக்கவும்'
    },
    'Tomato___Target_Spot': {
        'disease': 'தக்காளி இலக்கு புள்ளி நோய்',
        'cause': 'கோர்னஸ்போரா கேசிக்கோலா பூஞ்சை காரணமாக வருகிறது',
        'solution': 'அசோக்ஸிஸ்ட்ரோபின் பூஞ்சைக்கொல்லி தெளிக்கவும்',
        'prevention': 'பாதிக்கப்பட்ட இலைகளை அகற்றவும்'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'disease': 'தக்காளி மஞ்சள் இலை சுருள் வைரஸ்',
        'cause': 'வெள்ளை ஈ மூலம் பரவும் வைரஸ் காரணமாக வருகிறது',
        'solution': 'இமிடாக்லோப்ரிட் பூச்சிக்கொல்லி தெளிக்கவும்',
        'prevention': 'வெள்ளை ஈயை கட்டுப்படுத்தவும்'
    },
    'Tomato___Tomato_mosaic_virus': {
        'disease': 'தக்காளி மொசைக் வைரஸ்',
        'cause': 'டோபாமோவைரஸ் வைரஸ் காரணமாக வருகிறது',
        'solution': 'பாதிக்கப்பட்ட செடிகளை அகற்றவும்',
        'prevention': 'கருவிகளை கிருமிநாசினி மூலம் சுத்தம் செய்யவும்'
    },
    'Tomato___healthy': {
        'disease': 'தக்காளி ஆரோக்கியமான இலை ✅',
        'cause': 'நோய் எதுவும் இல்லை',
        'solution': 'தொடர்ந்து கண்காணிக்கவும்',
        'prevention': 'சரியான உரம் மற்றும் நீர் பராமரிக்கவும்'
    },
}


@app.get('/')
def health():
    return {
        'status': 'Plant Disease API Live! 🌿',
        'version': '3.0.0',
        'model': 'YOLOv8 ONNX + Gemini AI',
        'classes': 38,
        'language': 'Tamil'
    }


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

            # Step 1: Direct match
            info = disease_solutions.get(raw_name)

            # Step 2: Fuzzy match
            if not info:
                search_term = raw_name.lower().replace(
                    'leaf', '').replace('s', '').strip()
                search_words = search_term.split()
                for key, data in disease_solutions.items():
                    key_clean = key.lower().replace(
                        '___', ' ').replace('_', ' ')
                    if all(word in key_clean
                           for word in search_words):
                        info = data
                        break

            # Step 3: Default fallback
            if not info:
                info = {
                    'disease': raw_name.replace('_', ' '),
                    'cause': 'பூஞ்சை அல்லது வைரஸ் தொற்று',
                    'solution': 'சரியான பூஞ்சைக்கொல்லி தெளிக்கவும்',
                    'prevention': 'தோட்டத்தை சுத்தமாக வைத்திருங்கள்'
                }

            detections.append({
                'disease_name': info['disease'],
                'confidence': confidence,
                'cause': info['cause'],
                'solution': info['solution'],
                'prevention': info['prevention']
            })

    return {'detections': detections}


@app.post('/chat')
async def smart_farmer_chat(
        user_query: str = Body(..., embed=True)):
    try:
        system_message = SystemMessage(
            content=(
                "நீ ஒரு AI விவசாய நிபுணன். "
                "விவசாயிகளுக்கு தமிழிலும் ஆங்கிலத்திலும் "
                "பதில் சொல்வாய். பயிர் நோய்கள், சிகிச்சை, "
                "தடுப்பு முறைகள் பற்றி தெளிவாக விளக்குவாய்."
            )
        )
        user_message = HumanMessage(content=user_query)
        response = llm.invoke([system_message, user_message])
        return {"response": response.content}
    except Exception as e:
        return {
            "response": (
                "மன்னிக்கவும் நண்பா, "
                "Gemini இணைக்கப்படவில்லை. "
                "மீண்டும் முயற்சிக்கவும்."
            )
        }


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host='0.0.0.0', port=port)