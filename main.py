from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI(title='Plant Disease Detection API - 38 Classes')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load YOLOv8 model (best.onnx Render-la irukanum)
model = YOLO('best.onnx', task='detect')

# 38 Classes Master Dictionary
disease_solutions = {
    # APPLE
    'Apple___Apple_scab': {'disease': 'ஆப்பிள் செதில் நோய்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'கேப்டன் தெளிக்கவும்', 'prevention': 'இலைகளை அகற்றவும்'},
    'Apple___Black_rot': {'disease': 'ஆப்பிள் கருப்பு அழுகல்', 'cause': 'பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'கத்தரித்தல்'},
    'Apple___Cedar_apple_rust': {'disease': 'ஆப்பிள் துரு நோய்', 'cause': 'பூஞ்சை', 'solution': 'மைக்கோபுட்டானில்', 'prevention': 'ஜூனிபர் மரங்களை தவிர்க்கவும்'},
    'Apple___healthy': {'disease': 'ஆப்பிள் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிக்கவும்', 'prevention': 'கண்காணிக்கவும்'},
    
    # CHERRY, BLUEBERRY, RASPBERRY
    'Blueberry___healthy': {'disease': 'ப்ளூபெர்ரி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},
    'Cherry_(including_sour)___Powdery_mildew': {'disease': 'செர்ரி சாம்பல் நோய்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'கந்தக தூள்', 'prevention': 'ஈரப்பதம் குறைக்கவும்'},
    'Cherry_(including_sour)___healthy': {'disease': 'செர்ரி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},
    'Raspberry___healthy': {'disease': 'ராஸ்பெரி ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # CORN
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {'disease': 'சோளம் சாம்பல் புள்ளி', 'cause': 'செர்கோஸ்போரா பூஞ்சை', 'solution': 'பூஞ்சைக்கொல்லி', 'prevention': 'காற்றோட்டம்'},
    'Corn___Common_rust': {'disease': 'சோளம் துரு நோய்', 'cause': 'புக்கினியா பூஞ்சை', 'solution': 'ட்ரைஅசோல்', 'prevention': 'எதிர்ப்பு ரகங்கள்'},
    'Corn___Northern_Leaf_Blight': {'disease': 'சோளம் இலை கருகல்', 'cause': 'பூஞ்சை', 'solution': 'அசோக்ஸிஸ்ட்ரோபின்', 'prevention': 'பயிர் சுழற்சி'},
    'Corn___healthy': {'disease': 'சோளம் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # GRAPE
    'Grape___Black_rot': {'disease': 'திராட்சை கருப்பு அழுகல்', 'cause': 'பூஞ்சை', 'solution': 'மேன்கோசெப்', 'prevention': 'பழங்களை அகற்றவும்'},
    'Grape___Esca_(Black_Measles)': {'disease': 'திராட்சை எஸ்கா நோய்', 'cause': 'பூஞ்சை தொற்று', 'solution': 'கொடிகளை வெட்டவும்', 'prevention': 'கிருமிநாசினி'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'disease': 'திராட்சை இலை கருகல்', 'cause': 'பூஞ்சை', 'solution': 'செம்பு மருந்து', 'prevention': 'காற்றோட்டம்'},
    'Grape___healthy': {'disease': 'திராட்சை ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # PEACH, ORANGE, PEPPER
    'Orange___Haunglongbing_(Citrus_greening)': {'disease': 'ஆரஞ்சு பச்சையாதல்', 'cause': 'பாக்டீரியா', 'solution': 'ஆன்டிபயாடிக்', 'prevention': 'பூச்சி கட்டுப்பாடு'},
    'Peach___Bacterial_spot': {'disease': 'பீச் பாக்டீரியா புள்ளி', 'cause': 'பாக்டீரியா', 'solution': 'செம்பு ஹைட்ராக்சைடு', 'prevention': 'நல்ல கன்றுகள்'},
    'Peach___healthy': {'disease': 'பீச் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},
    'Pepper,_bell___Bacterial_spot': {'disease': 'மிளகாய் பாக்டீரியா புள்ளி', 'cause': 'பாக்டீரியா', 'solution': 'செம்பு மருந்து', 'prevention': 'நல்ல விதைகள்'},
    'Pepper,_bell___healthy': {'disease': 'மிளகாய் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # POTATO
    'Potato___Early_blight': {'disease': 'உருளைக்கிழங்கு முன் கருகல்', 'cause': 'ஆல்டர்னேரியா', 'solution': 'குளோரோதலோனில்', 'prevention': 'இடைவெளி'},
    'Potato___Late_blight': {'disease': 'உருளைக்கிழங்கு பின் கருகல்', 'cause': 'பைட்டோஃப்தோரா', 'solution': 'மேன்கோசெப்', 'prevention': 'வடிகால்'},
    'Potato___healthy': {'disease': 'உருளைக்கிழங்கு ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},

    # STRAWBERRY, SQUASH, SOYBEAN
    'Soybean___healthy': {'disease': 'சோயாபீன் ஆரோக்கியமானது ✅', 'cause': 'இல்லை', 'solution': 'பராமரிப்பு', 'prevention': 'கண்காணிப்பு'},
    'Squash___Powdery_mildew': {'disease': 'ஸ்குவாஷ் சாம்பல் நோய்', 'cause': 'பூஞ்சை', 'solution': 'கந்தக மருந்து', 'prevention': 'காற்றோட்டம்'},
    'Strawberry___Leaf_scorch': {'disease': 'ஸ்ட்ராபெர்ரி இலை கருகல்', 'cause': 'பூஞ்சை', 'solution': 'கேப்டன்', 'prevention': 'இலைகளை அகற்றவும்'},
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
}

@app.get('/')
def health():
    return {'status': 'Plant API Ready', 'classes': 38}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Model Inference
    results = model(img)
    detections = []

    for r in results:
        for box in r.boxes:
            raw_name = model.names[int(box.cls)]
            confidence = round(float(box.conf) * 100, 2)

            # CLEANUP LOGIC: Spaces/Underscore mismatch-ah handle panna
            # e.g., 'Tomato Early blight leaf' -> 'Tomato___Early_blight'
            lookup_key = raw_name.replace(' ', '___')
            
            # Master Lookup
            info = disease_solutions.get(raw_name) or disease_solutions.get(lookup_key, {
                'disease': raw_name.replace('___', ' ').replace('_', ' '),
                'cause': 'பூஞ்சை/வைரஸ் தொற்று',
                'solution': 'உரிய பூஞ்சைக்கொல்லி தெளிக்கவும்',
                'prevention': 'விவசாய நிபுணரை அணுகவும்'
            })

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