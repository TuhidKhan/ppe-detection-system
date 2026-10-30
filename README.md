# SafeSite AI - PPE Detection for Workplace Safety

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![Backend: Railway](https://img.shields.io/badge/Deploy-Railway-0B0D0E.svg)](https://railway.app/)
[![Frontend: Netlify](https://img.shields.io/badge/Deploy-Netlify-00C7B7.svg)](https://www.netlify.com/)

An automated computer vision and safety compliance auditing system for industrial environments, construction sites, and manufacturing plants. Built with **YOLOv8**, **FastAPI**, and a decoupled modern web interface.

---

## 1. Project Overview & Architecture

In high-risk workplaces, manual inspection cannot guarantee 100% continuous oversight. This system inspects visual imagery, detects protective equipment (helmets vs. unprotected heads), applies safety compliance rules, and generates actionable audit reports.

### System Architecture (Decoupled Cloud Architecture)

```
[ User Browser / Site Camera ]
               │
               ▼
┌───────────────────────────────┐
│     NETLIFY (Frontend UI)     │ ── Static HTML5, Modern CSS, Responsive JS
│  https://your-app.netlify.app │    (Drag-and-drop imagery, live compliance cards)
└──────────────┬────────────────┘
               │  HTTP POST multipart/form-data
               ▼
┌───────────────────────────────┐
│     RAILWAY (Backend API)     │ ── FastAPI + OpenCV Headless + PyTorch
│ https://your-api.railway.app  │    (Runs YOLOv8 inference & compliance engine)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│      YOLOv8 Neural Model      │ ── fine-tuned on 5,000 workplace safety images
│       (models/best.pt)        │    (Classes: Helmet [0], Head [1], Person [2])
└───────────────────────────────┘
```

---

## 2. Directory Structure (Strictly Compliant with Project Spec)

```text
PPE Detection_project_4/
├── app.py                     # Production FastAPI backend server
├── train.py                   # Model fine-tuning & evaluation script
├── data.yaml                  # YOLOv8 dataset configuration
├── requirements.txt           # Python dependencies
├── Procfile                   # Railway process execution command
├── README.md                  # System documentation
│
├── dataset/                   # Prepared YOLO dataset (5,000 items)
│   ├── images/
│   │   ├── train/             # 4,000 training photos (80%)
│   │   └── val/               # 1,000 validation photos (20%)
│   └── labels/
│       ├── train/             # Normalized YOLO .txt coordinates
│       └── val/               # Normalized YOLO .txt coordinates
│
├── models/                    # Model weights directory
│   └── best.pt                # Trained PPE detector weights
│
├── utils/                     # Modular business logic & utilities
│   ├── __init__.py
│   ├── converter.py           # Pascal VOC XML to YOLO TXT converter
│   ├── compliance.py          # Workplace safety rules engine
│   └── visualizer.py          # Bounding box & status badge drawing engine
│
└── frontend/                  # Netlify-ready client interface
    ├── index.html             # Professional, accessible layout (zero emojis)
    ├── style.css              # Industrial tech design system (responsive)
    └── script.js              # Reactive API caller & metrics renderer
```

---

## 3. Workplace Safety Compliance Logic

The system translates raw object detections into actionable safety statuses:

| Compliance Status | Condition | Action / Recommendation | Visual Indicator |
| :--- | :--- | :--- | :--- |
| **COMPLIANT** | `head == 0` and `helmet > 0` | All workers equipped with certified headgear. Authorized to work. | Green Badge |
| **VIOLATION** | `head > 0` | At least 1 unprotected worker detected. Immediate hazard. Access denied / Alert supervisor. | Red Alert Badge |
| **UNKNOWN** | `person > 0`, but head occluded | Worker body detected, but headgear is not clearly identifiable. | Amber Badge |
| **NO WORKERS** | `person == 0, helmet == 0, head == 0` | Vacant job site or unpopulated work zone. | Neutral Gray |

$$\text{Compliance Rate} = \left( \frac{\text{helmets}}{\text{helmets} + \text{unprotected heads}} \right) \times 100\%$$

---

## 4. Local Quickstart Guide

### Step 1: Clone and Install Dependencies
```bash
# Clone the repository
git clone <repository-url>
cd "PPE Detection_project_4"

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Start the FastAPI Backend
```bash
python app.py
```
The API is now running locally at: **`http://localhost:8000`**
Interactive Swagger documentation is available at: **`http://localhost:8000/docs`**

### Step 3: Launch the Frontend
You can open `frontend/index.html` directly in any web browser, or serve it using Python's built-in HTTP server:
```bash
cd frontend
python -m http.server 3000
```
Open **`http://localhost:3000`** in your browser.

---

## 5. Model Training (YOLOv8 Fine-Tuning)

To train or retrain the model on the 5,000 image dataset:

```bash
python train.py --epochs 35 --batch 16 --imgsz 640 --model yolov8n.pt
```

### Parameters:
* `--epochs`: Training iterations (default: 35).
* `--batch`: Batch size (default: 16).
* `--imgsz`: Input image resolution (default: 640).
* `--model`: Starting base checkpoint (`yolov8n.pt`).

When training completes, `train.py` automatically validates the model against the 1,000 validation images and copies the highest-scoring checkpoint to **`models/best.pt`**.

---

## 6. Deployment Guide

### A. Deploying the Backend on Railway

1. Sign in to [Railway.app](https://railway.app/).
2. Click **New Project** $\rightarrow$ **Deploy from GitHub Repo**.
3. Select this repository.
4. Railway will automatically detect:
   - `requirements.txt` (Installs Python dependencies)
   - `Procfile` (`web: uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}`)
5. In **Settings** $\rightarrow$ **Networking**, click **Generate Domain** (e.g. `https://ppe-backend-production.up.railway.app`).
6. Copy this domain URL.

### B. Deploying the Frontend on Netlify

1. Sign in to [Netlify.com](https://www.netlify.com/).
2. Drag and drop the **`frontend`** folder directly into the Netlify dashboard (or connect via GitHub repository with publish directory set to `frontend`).
3. Open your deployed Netlify website (e.g. `https://safesite-ppe.netlify.app`).
4. Click **API Settings** in the top navigation bar.
5. Paste your Railway backend URL (e.g. `https://ppe-backend-production.up.railway.app`) and click **Save & Test**.
6. The status indicator will turn green: **Backend: Operational**.

---

## 7. API Reference

### `POST /detect`
Accepts an image and returns compliance status and annotated imagery.

* **Content-Type:** `multipart/form-data`
* **Parameters:**
  * `file`: Binary image file (PNG, JPG, JPEG, WEBP)
  * `confidence_threshold` (query, optional): Confidence float `0.05` to `0.95` (default: `0.25`)
  * `include_banner` (query, optional): Boolean whether to render header status badge (default: `true`)

**Sample JSON Response:**
```json
{
  "success": true,
  "filename": "site_photo.jpg",
  "compliance": {
    "is_safe": false,
    "status": "VIOLATION",
    "status_level": "danger",
    "color_hex": "#EF4444",
    "message": "SAFETY ALERT: 1 worker(s) detected without safety helmets!",
    "compliance_rate": 66.7,
    "counts": {
      "helmet": 2,
      "head": 1,
      "person": 3,
      "total_heads_evaluated": 3
    }
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "helmet",
      "confidence": 0.942,
      "box": [120.5, 45.2, 185.0, 110.4]
    },
    {
      "class_id": 1,
      "class_name": "head",
      "confidence": 0.887,
      "box": [340.1, 52.0, 395.4, 118.2]
    }
  ],
  "total_detections": 2,
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
}
```

---

## 8. Author & Developer Credits

* **Developer:** **Tuhid Khan**
* **GitHub Profile:** [github.com/TuhidKhan](https://github.com/TuhidKhan)
* **LinkedIn Profile:** [linkedin.com/in/tuhid-khan](https://www.linkedin.com/in/tuhid-khan/)
* **Project Repository:** [github.com/TuhidKhan/ppe-detection-system](https://github.com/TuhidKhan/ppe-detection-system)

---

## 9. License & Acknowledgments
* **Base Architecture:** YOLOv8 by Ultralytics
* **Dataset:** Hard Hat Workers Dataset (Andrew MVD / Kaggle)
* **Frameworks:** FastAPI, OpenCV Headless, Modern HTML5/CSS3
