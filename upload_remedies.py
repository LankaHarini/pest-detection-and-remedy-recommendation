import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate(r"D:\WEB\Quantum Saved Extensions\APK File.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

remedies_data = {
    "fall armyworm": {
        "remedy": "Spray Emamectin Benzoate 5% SG",
        "dosage": "0.4g per liter",
        "prevention": "Use pheromone traps"
    },
    "leaf blight": {
        "remedy": "Spray Mancozeb 75% WP",
        "dosage": "2g per liter",
        "prevention": "Avoid overhead irrigation"
    },
    "leaf bettle": {
        "remedy": "Apply Neem Oil Spray",
        "dosage": "5ml per liter",
        "prevention": "Maintain field hygiene"
    },
    "grasshoper": {
        "remedy": "Use Malathion 50% EC",
        "dosage": "2ml per liter",
        "prevention": "Mechanical trapping"
    },
    "leaf spot": {
        "remedy": "Apply Carbendazim 50% WP",
        "dosage": "1g per liter",
        "prevention": "Use disease resistant seeds"
    },
    "streak virus": {
        "remedy": "Control vector insects using Imidacloprid",
        "dosage": "0.3ml per liter",
        "prevention": "Use certified virus-free seeds"
    },
    "healthy": {
        "remedy": "No treatment needed",
        "dosage": "-",
        "prevention": "Maintain proper irrigation"
    }
}

for disease, data in remedies_data.items():
    db.collection("remedies").document(disease).set(data)

print("Remedies uploaded successfully!")
