from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

disease_solutions = {
    "Tomato___Early_blight": {
        "disease": "Tomato Early Blight",
        "cause": "Alternaria solani fungus",
        "solution": "Copper-based fungicide spray pannunga. Infected leaves remove pannunga.",
        "prevention": "Crop rotation pannunga. Over-watering avoid pannunga."
    },
    "Tomato___Late_blight": {
        "disease": "Tomato Late Blight",
        "cause": "Phytophthora infestans",
        "solution": "Mancozeb fungicide use pannunga. Affected plants destroy pannunga.",
        "prevention": "Well-drained soil use pannunga. Air circulation maintain pannunga."
    },
    "Tomato___Bacterial_spot": {
        "disease": "Tomato Bacterial Spot",
        "cause": "Xanthomonas bacteria",
        "solution": "Copper bactericide spray pannunga. Infected leaves remove pannunga.",
        "prevention": "Seed treatment pannunga. Avoid overhead irrigation."
    },
    "Tomato___Target_Spot": {
        "disease": "Tomato Target Spot",
        "cause": "Corynespora cassiicola fungus",
        "solution": "Azoxystrobin fungicide spray pannunga.",
        "prevention": "Proper plant spacing maintain pannunga."
    },
    "Tomato___Tomato_mosaic_virus": {
        "disease": "Tomato Mosaic Virus",
        "cause": "Tobamovirus",
        "solution": "Infected plants remove pannunga. No chemical cure.",
        "prevention": "Virus-free seeds use pannunga. Tools sanitize pannunga."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease": "Tomato Yellow Leaf Curl Virus",
        "cause": "Begomovirus (whitefly transmitted)",
        "solution": "Infected plants remove pannunga. Whitefly control pannunga.",
        "prevention": "Resistant varieties use pannunga. Insecticide spray pannunga."
    },
    "Tomato___Leaf_Mold": {
        "disease": "Tomato Leaf Mold",
        "cause": "Passalora fulva fungus",
        "solution": "Chlorothalonil fungicide spray pannunga.",
        "prevention": "Humidity reduce pannunga. Good ventilation maintain pannunga."
    },
    "Tomato___Septoria_leaf_spot": {
        "disease": "Tomato Septoria Leaf Spot",
        "cause": "Septoria lycopersici fungus",
        "solution": "Mancozeb or copper fungicide spray pannunga.",
        "prevention": "Crop rotation pannunga. Mulching use pannunga."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease": "Tomato Spider Mites",
        "cause": "Tetranychus urticae mite",
        "solution": "Miticide or neem oil spray pannunga.",
        "prevention": "Regular water spray pannunga. Natural predators encourage pannunga."
    },
    "Tomato___healthy": {
        "disease": "Healthy Tomato",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Potato___Early_blight": {
        "disease": "Potato Early Blight",
        "cause": "Alternaria solani fungus",
        "solution": "Chlorothalonil fungicide use pannunga.",
        "prevention": "Crop rotation pannunga. Proper spacing maintain pannunga."
    },
    "Potato___Late_blight": {
        "disease": "Potato Late Blight",
        "cause": "Phytophthora infestans",
        "solution": "Metalaxyl fungicide spray pannunga. Infected plants burn pannunga.",
        "prevention": "Resistant varieties use pannunga."
    },
    "Potato___healthy": {
        "disease": "Healthy Potato",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Grape___Black_rot": {
        "disease": "Grape Black Rot",
        "cause": "Guignardia bidwellii fungus",
        "solution": "Myclobutanil fungicide spray pannunga. Infected fruits remove pannunga.",
        "prevention": "Pruning properly pannunga. Humidity control pannunga."
    },
    "Grape___Esca_(Black_Measles)": {
        "disease": "Grape Esca Black Measles",
        "cause": "Fungal complex (Phaeomoniella)",
        "solution": "Infected wood prune pannunga. Wound sealant use pannunga.",
        "prevention": "Pruning wounds protect pannunga."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "disease": "Grape Leaf Blight",
        "cause": "Isariopsis clavispora fungus",
        "solution": "Copper fungicide spray pannunga.",
        "prevention": "Good air circulation maintain pannunga."
    },
    "Grape___healthy": {
        "disease": "Healthy Grape",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Strawberry___Leaf_scorch": {
        "disease": "Strawberry Leaf Scorch",
        "cause": "Diplocarpon earlianum fungus",
        "solution": "Captan fungicide spray pannunga. Old leaves remove pannunga.",
        "prevention": "Certified disease-free plants use pannunga."
    },
    "Strawberry___healthy": {
        "disease": "Healthy Strawberry",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Corn_(maize)___Common_rust_": {
        "disease": "Corn Common Rust",
        "cause": "Puccinia sorghi fungus",
        "solution": "Triazole fungicide spray pannunga.",
        "prevention": "Resistant hybrid seeds use pannunga."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "disease": "Corn Northern Leaf Blight",
        "cause": "Exserohilum turcicum fungus",
        "solution": "Propiconazole fungicide spray pannunga.",
        "prevention": "Resistant varieties use pannunga. Crop rotation pannunga."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "disease": "Corn Gray Leaf Spot",
        "cause": "Cercospora zeae-maydis fungus",
        "solution": "Strobilurin fungicide spray pannunga.",
        "prevention": "Crop rotation pannunga. Tillage practices follow pannunga."
    },
    "Corn_(maize)___healthy": {
        "disease": "Healthy Corn",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "cause": "Venturia inaequalis fungus",
        "solution": "Captan or mancozeb fungicide spray pannunga.",
        "prevention": "Fallen leaves destroy pannunga. Resistant varieties use pannunga."
    },
    "Apple___Black_rot": {
        "disease": "Apple Black Rot",
        "cause": "Botryosphaeria obtusa fungus",
        "solution": "Captan fungicide spray pannunga. Infected branches prune pannunga.",
        "prevention": "Dead wood remove pannunga regularly."
    },
    "Apple___Cedar_apple_rust": {
        "disease": "Apple Cedar Rust",
        "cause": "Gymnosporangium juniperi-virginianae",
        "solution": "Myclobutanil fungicide spray pannunga.",
        "prevention": "Nearby juniper trees remove pannunga."
    },
    "Apple___healthy": {
        "disease": "Healthy Apple",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "disease": "Cherry Powdery Mildew",
        "cause": "Podosphaera clandestina fungus",
        "solution": "Sulfur or potassium bicarbonate spray pannunga.",
        "prevention": "Good air circulation maintain pannunga."
    },
    "Cherry_(including_sour)___healthy": {
        "disease": "Healthy Cherry",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Peach___Bacterial_spot": {
        "disease": "Peach Bacterial Spot",
        "cause": "Xanthomonas arboricola bacteria",
        "solution": "Copper bactericide spray pannunga.",
        "prevention": "Resistant varieties use pannunga."
    },
    "Peach___healthy": {
        "disease": "Healthy Peach",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Pepper,_bell___Bacterial_spot": {
        "disease": "Pepper Bacterial Spot",
        "cause": "Xanthomonas campestris bacteria",
        "solution": "Copper fungicide spray pannunga. Infected plants remove pannunga.",
        "prevention": "Certified seeds use pannunga."
    },
    "Pepper,_bell___healthy": {
        "disease": "Healthy Pepper",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Squash___Powdery_mildew": {
        "disease": "Squash Powdery Mildew",
        "cause": "Podosphaera xanthii fungus",
        "solution": "Neem oil or sulfur spray pannunga.",
        "prevention": "Proper spacing maintain pannunga."
    },
    "Raspberry___healthy": {
        "disease": "Healthy Raspberry",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Soybean___healthy": {
        "disease": "Healthy Soybean",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "disease": "Citrus Greening HLB",
        "cause": "Candidatus Liberibacter bacteria",
        "solution": "No cure. Infected trees remove pannunga.",
        "prevention": "Psyllid insect control pannunga. Certified trees use pannunga."
    },
    "Blueberry___healthy": {
        "disease": "Healthy Blueberry",
        "cause": "No disease",
        "solution": "No treatment needed!",
        "prevention": "Regular monitoring continue pannunga."
    },
}

@app.get("/")
def health():
    return {"status": "Plant Disease API running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model(img)
    detections = []
    for r in results:
        for box in r.boxes:
            disease_key = model.names[int(box.cls)]
            solution = disease_solutions.get(disease_key, {
                "disease": disease_key.replace("___", " - ").replace("_", " "),
                "cause": "Further analysis needed",
                "solution": "Consult an agricultural expert",
                "prevention": "Regular monitoring pannunga"
            })
            detections.append({
                "disease_key": disease_key,
                "disease_name": solution["disease"],
                "confidence": round(float(box.conf) * 100, 2),
                "cause": solution["cause"],
                "solution": solution["solution"],
                "prevention": solution["prevention"],
                "bbox": box.xyxy[0].tolist()
            })
    return {"detections": detections, "total": len(detections)}