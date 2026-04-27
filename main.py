from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI(title='Plant Disease Detection API - 38 Classes Fixed')

# CORS setup for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load YOLOv8 model (best.onnx)
model = YOLO('best.onnx', task='detect')

# 38 Classes Master Dictionary with Tamil Mapping
disease_solutions = {
    # APPLE
    'Apple___Apple_scab': {'disease': 'ஆப்பிள் செதில் நோய்', 'cause': 'வென்டுரியா பூஞ்சை', 'solution': 'கேப்டன் தெளிக்கவும்', 'prevention': 'இலைகளை அகற்றவும்'},
    'Apple___Black_rot': {'disease': 'ஆப்பிள் கருப்பு அழுகல்', 'cause': 'போட்ரியோஸ்பேரியா பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'கத்தரித்தல்'},
    'Apple___Cedar_apple_rust': {'disease': 'ஆப்பிள் துரு நோய்', 'cause': 'ஜிம்னோஸ்போரான்ஜியம்', 'solution': 'மைக்கோபுட்டானில்', 'prevention': 'ஜூனிபர் மரங்களை தவிர்க்கவும்'},
    'Apple___healthy': {'disease': 'ஆப்பிள் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிக்கவும்', 'prevention': 'கண்காணிக்கவும்'},
    
    # CORN
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {'disease': 'சோளம் சாம்பல் புள்ளி', 'cause': 'செர்கோஸ்போரா பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'காற்றோட்டம்'},
    'Corn___Common_rust': {'disease': 'சோளம் துரு நோய்', 'cause': 'புக்கினியா பூஞ்சை', 'solution': 'ட்ரைஅசோல்', 'prevention': 'எதிர்ப்பு ரகங்கள்'},
    'Corn___Northern_Leaf_Blight': {'disease': 'சோளம் இலை கருகல்', 'cause': 'பூஞ்சை', 'solution': 'அசோக்ஸிஸ்ட்ரோபின்', 'prevention': 'பயிர் சுழற்சி'},
    'Corn___healthy': {'disease': 'சோளம் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # POTATO
    'Potato___Early_blight': {'disease': 'உருளைக்கிழங்கு முன் கருகல்', 'cause': 'ஆல்டர்னேரியா', 'solution': 'குளோரோதலோனில்', 'prevention': 'இடைவெளி'},
    'Potato___Late_blight': {'disease': 'உருளைக்கிழங்கு பின் கருகல்', 'cause': 'பைட்டோஃப்தோரா', 'solution': 'மேன்கோசெப்', 'prevention': 'வடிகால்'},
    'Potato___healthy': {'disease': 'உருளைக்கிழங்கு ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # STRAWBERRY
    'Strawberry___Leaf_scorch': {'disease': 'ஸ்ட்ராபெர்ரி இலை கருகல்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'கேப்டன்', 'prevention': 'இலைகளை அகற்றவும்'},
    'Strawberry___healthy': {'disease': 'ஸ்ட்ராபெர்ரி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # TOMATO
    'Tomato___Bacterial_spot': {'disease': 'தக்காளி பாக்டீரியா புள்ளி', 'cause': 'பாக்டீரியா', 'solution': 'செம்பு சல்பேட்', 'prevention': 'கிருமிநாசினி'},
    'Tomato___Early_blight': {'disease': 'தக்காளி ஆரம்ப கருகல்', 'cause': 'ஆல்டர்னேரியா', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'பயிர் சுழற்சி'},
    'Tomato___Late_blight': {'disease': 'தக்காளி தாமத கருகல்', 'cause': 'பைட்டோஃப்தோரா', 'solution': 'மேன்கோசெப்', 'prevention': 'வடிகால்'},
    'Tomato___Leaf_Mold': {'disease': 'தக்காளி இலை அச்சு நோய்', 'cause': 'பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'காற்றோட்டம்'},
    'Tomato___Septoria_leaf_spot': {'disease': 'தக்காளி செப்டோரியா புள்ளி', 'cause': 'பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'இலை நீக்கம்'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'disease': 'தக்காளி சிலந்தி பூச்சி', 'cause': 'சிலந்தி தாக்குதல்', 'solution': 'அபாமெக்டின்', 'prevention': 'கண்காணிப்பு'},
    'Tomato___Target_Spot': {'disease': 'தக்காளி இலக்கு புள்ளி', 'cause': 'பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'இலைகளை அகற்றவும்'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'disease': 'தக்காளி மஞ்சள் வைரஸ்', 'cause': 'வெள்ளை ஈ', 'solution': 'இமிடாக்லோப்ரிட்', 'prevention': 'ஈக்களை கட்டுப்படுத்தவும்'},
    'Tomato___Tomato_mosaic_virus': {'disease': 'தக்காளி மொசைக் வைரஸ்', 'cause': 'வைரஸ்', 'solution': 'செடியை அகற்றவும்', 'prevention': 'சுத்தம்'},
    'Tomato___healthy': {'disease': 'தக்காளி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},
    
    # OTHERS (Grape, Orange, Peach, Pepper etc - detailed solutions follow same pattern)
    'Orange___Haunglongbing_(Citrus_greening)': {'disease': 'ஆரஞ்சு பச்சையாதல்', 'cause': 'பாக்டீரியா', 'solution': 'ஆன்டிபயாடிக்', 'prevention': 'பூச்சி கட்டுப்பாடு'},
    'Grape___Black_rot': {'disease': 'திராட்சை கருப்பு அழுகல்', 'cause': 'பூஞ்சை', 'solution': 'மேன்கோசெப்', 'prevention': 'பழங்களை அகற்றவும்'},
}

@app.get('/')
def health():
    return {'status': 'Plant API Ready', 'classes': 38}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    results = model(img)
    detections = []

    for r in results:
        for box in r.boxes:
            raw_name = model.names[int(box.cls)] # e.g., "Tomato leaf late blights"
            confidence = round(float(box.conf) * 100, 2)

            # --- DYNAMIC FUZZY MATCHING LOGIC ---
            info = None
            
            # Clean raw name: "Tomato leaf late blights" -> "tomato late blight"
            clean_raw = raw_name.lower().replace('leaf', '').replace('leafs', '').replace('blights', 'blight').strip()
            clean_raw = " ".join(clean_raw.split())

            # Dictionary-la search pannuvom
            for key, data in disease_solutions.items():
                # Dictionary key clean-up: "Tomato___Late_blight" -> "tomato late blight"
                clean_key = key.lower().replace('___', ' ').replace('_', ' ').replace('leaf', '').strip()
                clean_key = " ".join(clean_key.split())

                if clean_raw == clean_key or clean_key in clean_raw or clean_raw in clean_key:
                    info = data
                    break

            # If still no match, fallback to raw name
            if not info:
                info = {
                    'disease': raw_name.replace('___', ' ').replace('_', ' '),
                    'cause': 'பூஞ்சை அல்லது வைரஸ் தொற்று',
                    'solution': 'உரிய பூஞ்சைக்கொல்லி தெளிக்கவும்',
                    'prevention': 'விவசாய நிபுணரை அணுகவும்'
                }

            detections.append({
                'disease_name': info['disease'],
                'confidence': confidence,
                'cause': info['cause'],
                'solution': info['solution'],
                'prevention': info['prevention']
            })

    return {'detections': detections}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run('main:app', host='0.0.0.0', port=port)