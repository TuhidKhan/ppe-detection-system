"""
==============================================================================
PPE Detection Project - Production Backend API (FastAPI)
==============================================================================
Designed for deployment on Railway (or any cloud container) with CORS enabled
to connect seamlessly with a Netlify frontend.

API Endpoints:
  - GET  /        : API overview and health status
  - GET  /health  : Automated Railway container health check
  - POST /detect  : Accepts worker image upload, runs YOLOv8 PPE detection,
                    evaluates safety compliance, and returns both JSON report
                    and base64-encoded annotated image.
==============================================================================
"""

import os
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Import our custom modular utilities
from utils.compliance import evaluate_compliance
from utils.visualizer import (
    draw_bounding_boxes,
    draw_status_banner,
    encode_image_to_base64,
    decode_image_bytes
)

# 1. Initialize FastAPI Application
app = FastAPI(
    title="PPE Safety Detection API",
    description="Automated AI safety compliance system for construction and industrial environments.",
    version="1.0.0"
)

# 2. Configure Cross-Origin Resource Sharing (CORS)
# This allows our Netlify frontend (or any client domain) to make requests to this API without browser CORS blocks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # In production, can restrict to your Netlify URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Model Loading Logic
# Priority:
#   1. models/best.pt (Our custom trained PPE model)
#   2. yolov8n.pt     (Fallback base model if training is in progress)
MODEL_PATH = Path("models/best.pt")
FALLBACK_MODEL = "yolov8n.pt"

try:
    if MODEL_PATH.exists():
        print(f"[Model Loader] Loading custom trained PPE model from: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
    else:
        print(f"[Model Loader] Custom model '{MODEL_PATH}' not found yet. Using base model '{FALLBACK_MODEL}' as fallback.")
        model = YOLO(FALLBACK_MODEL)
except Exception as e:
    print(f"[Model Loader Error] Could not load model: {e}")
    model = None


@app.get("/")
def root_info() -> Dict[str, Any]:
    """
    Root endpoint providing system health, model status, and API instructions.
    """
    classes = list(model.names.values()) if model else []
    return {
        "project": "PPE Detection for Workplace Safety",
        "status": "Online",
        "model_loaded": model is not None,
        "model_file": str(MODEL_PATH) if MODEL_PATH.exists() else FALLBACK_MODEL,
        "model_classes": classes,
        "is_fine_tuned": "helmet" in classes,
        "docs_url": "/docs",
        "detect_endpoint": "/detect (POST multipart/form-data with 'file')"
    }


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Railway Health Probe Endpoint.
    Cloud platforms query this endpoint periodically to verify the container is alive.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="AI Model is not loaded.")
    classes = list(model.names.values())
    return {
        "status": "healthy",
        "service": "ppe-detection-backend",
        "classes": classes,
        "is_fine_tuned": "helmet" in classes
    }


@app.post("/detect")
async def detect_ppe(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(
        0.25,
        ge=0.05,
        le=0.95,
        description="Minimum confidence score threshold for detections (0.05 - 0.95)"
    ),
    include_banner: bool = Query(
        True,
        description="Whether to render the compliance banner across the top of the image"
    )
) -> Dict[str, Any]:
    """
    Main detection endpoint:
      1. Receives an uploaded worker image.
      2. Decodes image into OpenCV format.
      3. Runs YOLOv8 inference to find helmets, bare heads, and workers.
      4. Evaluates safety compliance rules.
      5. Annotates the image with colored bounding boxes and status banner.
      6. Returns structured JSON report + base64 image string for instant frontend display.
    """
    # Validate file format
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file '{file.filename}' is not a valid image. (Content-Type: {file.content_type})"
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="AI Detection model is currently unavailable."
        )

    try:
        # Read file bytes from HTTP upload stream
        image_bytes = await file.read()

        # Decode into OpenCV numpy array (BGR)
        cv_image = decode_image_bytes(image_bytes)

        # Run YOLOv8 object detection inference
        results = model.predict(
            source=cv_image,
            conf=confidence_threshold,
            verbose=False
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]

                # Map class name using model's names dictionary
                class_name = result.names.get(cls_id, f"class_{cls_id}").lower()

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "box": coords  # [xmin, ymin, xmax, ymax]
                })

        # Evaluate workplace safety compliance rules
        compliance_report = evaluate_compliance(detections)

        # Draw bounding boxes on image
        annotated_image = draw_bounding_boxes(cv_image, detections)

        # Optionally draw the top compliance banner
        if include_banner:
            annotated_image = draw_status_banner(annotated_image, compliance_report)

        # Encode annotated image to Base64 data URI string for the frontend
        base64_image_uri = encode_image_to_base64(annotated_image, format_ext=".jpg")

        return {
            "success": True,
            "filename": file.filename,
            "compliance": compliance_report,
            "detections": detections,
            "total_detections": len(detections),
            "annotated_image": base64_image_uri
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during PPE detection: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Railway automatically injects the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting PPE Detection Backend on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
