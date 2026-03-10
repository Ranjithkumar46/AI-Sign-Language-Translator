# 🤟 AI-Powered Sign Language Translator

> **UG Final Year Project** — A real-time bidirectional sign language translation system using Deep Learning, Computer Vision, and 3D Animation.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-green?logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Detection-orange)
![Three.js](https://img.shields.io/badge/Three.js-3D_Animation-black?logo=three.js)
![License](https://img.shields.io/badge/License-Educational-yellow)

---

## 📌 About

This project translates **sign language gestures to text** using a live webcam, and converts **text back to 3D animated sign language** using Three.js. It uses a trained neural network (MLP) with MediaPipe hand landmark detection.

### Features

- ✅ **Sign → Text**: Real-time webcam gesture recognition (A-Z, 0-9, common words)
- ✅ **Text → Sign**: 3D animated hand performing sign gestures
- ✅ **44 Sign Classes**: 26 alphabets + 10 numbers + 8 common words
- ✅ **Auto Model Training**: Trains automatically on first run
- ✅ **Offline Support**: Works without internet after setup
- ✅ **Single File Backend**: Entire server in one `app.py`

### Common Words Supported

| Word | Word | Word | Word |
|------|------|------|------|
| Hello | Hi | Good Morning | Thank You |
| Sorry | Please | Yes | No |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| AI Model | scikit-learn MLPClassifier (512→256→128→64) |
| Hand Detection | Google MediaPipe Hands |
| Computer Vision | OpenCV |
| 3D Animation | Three.js (WebGL) |
| Frontend | HTML5, CSS3, JavaScript |
| Data Processing | NumPy, scikit-learn |

---

## 📁 Project Structure

```
AI-Sign-Language-Translator/
├── app.py                    # Complete backend (single file)
├── requirements.txt          # Python dependencies
├── .gitignore
├── README.md
├── templates/
│   ├── index.html           # Home page
│   ├── sign_to_text.html    # Webcam sign detection page
│   ├── text_to_sign.html    # 3D animation page
│   └── about.html           # Project info page
├── dataset/                  # Auto-generated on first run
│   └── landmark_dataset.npz
└── model/                    # Auto-generated on first run
    ├── sign_model.pkl
    └── label_encoder.pkl
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.9 or higher
- Webcam (for Sign → Text)
- Modern web browser

### Step 1: Clone this repository

```bash
git clone https://github.com/Ranjithkumar46/AI-Sign-Language-Translator
cd AI-Sign-Language-Translator
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the application

```bash
python app.py
```

### Step 4: Open in browser

```
http://localhost:5000
```

> On first run, the model will auto-train (~1-2 minutes). After that, it loads instantly.

---

## ⚙️ How It Works

### Sign → Text Pipeline

```
Webcam → MediaPipe Hand Detection → 21 Landmarks (63 features)
→ MLP Neural Network → Prediction → Sentence Builder → Display Text
```

### Text → Sign Pipeline

```
User Types Text → Check Common Words Database
→ If found: Play full word gesture animation
→ If not found: Finger-spell letter by letter
→ Three.js 3D Hand Animation
```

---

## 🧠 AI Model Details

- **Architecture**: Multi-Layer Perceptron (MLP)
- **Hidden Layers**: 512 → 256 → 128 → 64 neurons
- **Activation**: ReLU
- **Training Data**: 26,400 augmented samples (600 per class)
- **Augmentation**: Random noise, scaling, rotation, translation
- **Validation Accuracy**: ~86%

---

## 📸 Screenshots

> Add your screenshots here after running the project

---

## 📄 License

This project is created for educational purposes as a UG Final Year Project.

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for hand landmark detection
- [Three.js](https://threejs.org/) for 3D WebGL rendering
- [Flask](https://flask.palletsprojects.com/) for web framework
- [scikit-learn](https://scikit-learn.org/) for ML model
