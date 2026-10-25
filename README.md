# 🌿 DetectPlant: AI-Powered Plant Identification & Crop Disease Diagnosis

[![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B?style=for-the-badge&logo=flutter&logoColor=white)](https://flutter.dev/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![Architecture](https://img.shields.io/badge/Architecture-Feature--First_%2B_Riverpod-blueviolet?style=for-the-badge)](https://riverpod.dev/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**DetectPlant** is a high-performance, offline-first mobile application built with Flutter that puts botanical identification and agronomic diagnosis directly into farmers', agronomists', and gardening enthusiasts' hands. Powered by dual lightweight **EfficientNetB0 INT8-quantized TensorFlow Lite** models, the app delivers accurate real-time inference on-device without requiring an internet connection.

---

## 📑 Table of Contents

- [Key Highlights](#-key-highlights)
- [System Architecture](#-system-architecture)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
  - [Plant Identification Model](#1-plant-identification-model)
  - [Crop Disease Diagnosis Model](#2-crop-disease-diagnosis-model)
  - [Tensor Specifications & Preprocessing](#3-tensor-specifications--preprocessing)
- [Application Features](#-application-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Building APK via GitHub Actions (CI)](#-building-apk-via-github-actions-ci)
- [Verification & Python Testing](#-verification--python-testing)
- [Database Schema](#-database-schema)
- [Localization (10 Languages)](#-localization-10-languages)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Key Highlights

- **⚡ Dual On-Device Neural Networks**:
  - **Plant Identifier**: Classifies **47 ornamental and houseplant species**.
  - **Disease Detector**: Identifies **15 crop disease and health conditions** across Tomato, Potato, and Pepper crops.
- **🛡️ 100% Offline Inference**: Real-time TFLite classification operates completely offline with latency under 150ms.
- **🩺 Smart Condition Recognition**: Automatically recognizes healthy foliage (`Potato___healthy`, `Tomato_healthy`, `Pepper__bell___healthy`) and adjusts health status and risk calculations accordingly.
- **🌐 Dynamic Knowledge Lookup**: Integrates the Wikipedia REST API and DuckDuckGo fallback to fetch symptoms, causes, and treatment guidelines automatically.
- **📊 Diagnostic PDF Reports**: Generates professional multi-page downloadable PDF reports featuring timestamped scans, confidence metrics, visual preview, and recommended agronomic actions.
- **🌍 10 Native Languages**: Fully internationalized with live language switching (English, Spanish, Hindi, French, Arabic, Bengali, Russian, Portuguese, Indonesian, German).
- **📦 Cloud CI/CD Pipeline**: GitHub Actions workflow preconfigured to build, package, and archive signed release APKs on demand.

---

## 🏛️ System Architecture

DetectPlant follows a **Feature-First Layered Architecture** with **Riverpod** reactive state management:

```
┌────────────────────────────────────────────────────────┐
│                      UI Layer                          │
│   Dashboard  │  Scanner / Camera  │  History  │ Reports│
└───────────────────────────┬────────────────────────────┘
                            │ (Riverpod Providers)
┌───────────────────────────▼────────────────────────────┐
│                    Service Layer                       │
│  TFLiteService  │ WikipediaService │  PdfService       │
└─────────────┬─────────────────────┬──────────────┬─────┘
              │                     │              │
┌─────────────▼──────┐ ┌────────────▼──────┐ ┌─────▼─────┐
│  TFLite / C-API    │ │ Wikipedia / Web   │ │  SQLite   │
│  (Neural Engines)  │ │ (Async Fallbacks) │ │ (sqflite) │
└────────────────────┘ └───────────────────┘ └───────────┘
```

---

## 🧠 Machine Learning Pipeline

The application incorporates two separate, independently trained transfer learning models based on **EfficientNetB0** (`weights='imagenet'`), fine-tuned and quantized to INT8 precision for mobile edge deployment.

### 1. Plant Identification Model
- **Target Domain**: 47 indoor and ornamental houseplant species.
- **Classes**: African Violet, Aloe Vera, Boston Fern, Calathea, Fiddle Leaf Fig, Monstera, Pothos, Snake Plant, etc.
- **Weights File**: `assets/model/plant_model_quantized.tflite`
- **Labels File**: `assets/labels/plant_labels.txt`

### 2. Crop Disease Diagnosis Model
- **Target Domain**: PlantVillage agricultural disease subset.
- **Validation Accuracy**: **~93.3%** on independent test split.
- **Weights File**: `assets/model/disease_model_quantized.tflite`
- **Labels File**: `assets/labels/disease_labels.txt`

#### Diagnosable Conditions (15 Classes):
| Crop | Class Label | Condition / Pathogen |
| :--- | :--- | :--- |
| **Pepper (Bell)** | `Pepper__bell___Bacterial_spot` | Bacterial Spot (*Xanthomonas*) |
| **Pepper (Bell)** | `Pepper__bell___healthy` | Healthy Foliage (Control) |
| **Potato** | `Potato___Early_blight` | Early Blight (*Alternaria solani*) |
| **Potato** | `Potato___Late_blight` | Late Blight (*Phytophthora infestans*) |
| **Potato** | `Potato___healthy` | Healthy Foliage (Control) |
| **Tomato** | `Tomato_Bacterial_spot` | Bacterial Spot (*Xanthomonas*) |
| **Tomato** | `Tomato_Early_blight` | Early Blight (*Alternaria solani*) |
| **Tomato** | `Tomato_Late_blight` | Late Blight (*Phytophthora infestans*) |
| **Tomato** | `Tomato_Leaf_Mold` | Leaf Mold (*Passalora fulva*) |
| **Tomato** | `Tomato_Septoria_leaf_spot` | Septoria Leaf Spot (*Septoria lycopersici*) |
| **Tomato** | `Tomato_Spider_mites_Two_spotted_spider_mite` | Two-Spotted Spider Mite infestation |
| **Tomato** | `Tomato__Target_Spot` | Target Spot (*Corynespora cassiicola*) |
| **Tomato** | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Yellow Leaf Curl Virus (TYLCV) |
| **Tomato** | `Tomato__Tomato_mosaic_virus` | Tomato Mosaic Virus (ToMV) |
| **Tomato** | `Tomato_healthy` | Healthy Foliage (Control) |

### 3. Tensor Specifications & Preprocessing

- **Input Dimension**: `[1, 224, 224, 3]` (Batch, Height, Width, RGB Channels).
- **Data Type**: `Float32` (converted via integer quantized graph).
- **Scaling**: Raw $[0.0, 255.0]$ pixel floats. EfficientNetB0 contains internal scaling layers (`Rescaling(1./255)`), ensuring Flutter does not perform duplicate normalization.
- **Output Dimension**:
  - Plant Model: `[1, 47]` probabilities
  - Disease Model: `[1, 15]` probabilities

---

## 📱 Application Features

| Screen / Feature | Description |
| :--- | :--- |
| **Interactive Dashboard** | Quick-action navigation for Camera scan, Photo Picker, History logs, PDF generation, and Language Settings. |
| **Dual Scanner View** | Toggle between **Identify Plant** and **Detect Disease** before capturing or picking images. |
| **Diagnostic Result Screen** | Displays primary prediction with formatted labels (e.g. `TOMATO — EARLY BLIGHT`), confidence progress bar, risk level badge, and parsed Wikipedia treatment guide. |
| **PDF Report Generator** | Creates standardized A4 agronomic summary documents with embedded scan photos, diagnosis timestamps, and actionable notes. |
| **History Log (SQLite)** | Persistent records of previous scans with visual thumbnail caching, date-sorting, and detail review. |

---

## 📂 Project Structure

```text
DetectPlant/
├── .github/
│   └── workflows/
│       └── build_apk.yml           # Automated CI/CD release APK builder
├── android/                        # Native Android Gradle configuration
│   ├── app/
│   │   ├── build.gradle.kts       # Android app configuration (SDK 34, Java 17)
│   │   └── src/main/AndroidManifest.xml
│   └── gradle.properties          # JVM args & validation mode bypass
├── assets/
│   ├── labels/
│   │   ├── disease_labels.txt     # 15 crop disease category strings
│   │   └── plant_labels.txt       # 47 houseplant species strings
│   ├── model/
│   │   ├── disease_model_quantized.tflite   # Retrained 15-class disease model
│   │   └── plant_model_quantized.tflite     # 47-class plant model
│   └── translations/              # 10 JSON translation dictionaries
├── lib/
│   ├── core/
│   │   └── services/
│   │       ├── database_helper.dart  # SQLite CRUD operations
│   │       ├── pdf_service.dart      # PDF report construction
│   │       ├── tflite_service.dart   # Image preprocessing & TFLite inference
│   │       └── wikipedia_service.dart# API client for disease & plant info
│   ├── features/
│   │   ├── complaints/               # Feedback collection UI
│   │   ├── dashboard/                # Main menu dashboard
│   │   ├── history/                  # Scan logs review
│   │   ├── reports/                  # PDF export viewer
│   │   └── scanner/
│   │       ├── camera_screen.dart    # Live camera capture
│   │       └── result_screen.dart    # Diagnosis UI & treatment display
│   └── main.dart                     # App entry point, Riverpod, & EasyLocalization
├── test_disease_model.py             # Desktop CLI verification script
├── pubspec.yaml                      # Dependencies & asset declarations
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- [Flutter SDK](https://docs.flutter.dev/get-started/install) (`>= 3.22.0`)
- [Java Development Kit (JDK)](https://adoptium.net/) (Version 17)
- [Android Studio](https://developer.android.com/studio) or Android Command Line Tools

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/codxabhay2004/DetectPlant.git
   cd DetectPlant
   ```

2. **Install Flutter packages**:
   ```bash
   flutter pub get
   ```

3. **Verify Connected Device**:
   ```bash
   flutter devices
   ```

4. **Run in Debug Mode**:
   ```bash
   flutter run
   ```

5. **Build Release APK locally**:
   ```bash
   flutter build apk --release --no-tree-shake-icons
   ```
   The compiled APK will be at `build/app/outputs/flutter-apk/app-release.apk`.

---

### ☁️ Building APK via GitHub Actions (CI)

This repository includes a ready-to-run GitHub Actions workflow (`.github/workflows/build_apk.yml`):

1. Push any commit to `main` (or click **Run workflow** under the **Actions** tab on GitHub).
2. The workflow automatically provisions an Ubuntu runner with **Java 17**, installs **Flutter Stable**, runs `flutter pub get`, and executes `flutter build apk --release`.
3. When the workflow completes with a green checkmark, download the **`DetectPlant-Release-APK`** artifact zip directly from the job summary page.

---

## 🧪 Verification & Python Testing

You can verify the TFLite disease model directly on a desktop environment using Python before deploying to a physical device:

```bash
# 1. Run synthetic verification tensor
python test_disease_model.py

# 2. Run inference on a real leaf image
python test_disease_model.py "C:\path\to\leaf_photo.jpg"
```

**Sample Output**:
```text
Loading TFLite model from: assets/model/disease_model_quantized.tflite
Model Tensor Specifications:
  Input  shape : [  1 224 224   3] (float32)
  Output shape : [ 1 15] (float32)

--- Model Output (Top 5 Predictions) ---
  #1: Tomato_Late_blight                        97.66%
  #2: Tomato_Leaf_Mold                           1.56%
  #3: Potato___Early_blight                      0.39%
  #4: Tomato__Tomato_YellowLeaf__Curl_Virus      0.00%
  #5: Tomato__Target_Spot                        0.00%

Result Summary:
  Primary Prediction: Tomato_Late_blight
  Status            : Diseased
  Confidence        : 97.66%
[SUCCESS] Model executed inference successfully!
```

---

## 🗄️ Database Schema

The app uses **SQLite** through `sqflite` for local persistence:

```sql
CREATE TABLE scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  imagePath TEXT,
  plantDetected TEXT,
  diseaseDetected TEXT,
  diseaseConfidence TEXT,
  isHealthy INTEGER,
  scanDate TEXT
);

CREATE TABLE complaints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  email TEXT,
  message TEXT,
  date TEXT
);
```

---

## 🌐 Localization (10 Languages)

Supported languages configured in `assets/translations/`:
- 🇬🇧 **English (`en.json`)**
- 🇪🇸 **Spanish (`es.json`)**
- 🇮🇳 **Hindi (`hi.json`)**
- 🇫🇷 **French (`fr.json`)**
- 🇸🇦 **Arabic (`ar.json`)**
- 🇧🇩 **Bengali (`bn.json`)**
- 🇷🇺 **Russian (`ru.json`)**
- 🇧🇷 **Portuguese (`pt.json`)**
- 🇮🇩 **Indonesian (`id.json`)**
- 🇩🇪 **German (`de.json`)**

Switching languages in the settings immediately updates UI strings, formatted headers, and diagnostic results.

---

## 🤝 Contributing

1. Fork the Project.
2. Create your Feature Branch:
   ```bash
   git checkout -b feature/NewFeature
   ```
3. Commit your Changes:
   ```bash
   git commit -m "Add NewFeature"
   ```
4. Push to the Branch:
   ```bash
   git push origin feature/NewFeature
   ```
5. Open a Pull Request.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.
