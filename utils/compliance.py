"""
==============================================================================
PPE Detection Project - Safety Compliance Rules Engine
==============================================================================
This module translates raw AI object detections (bounding boxes + class labels)
into actionable safety decisions for job site supervisors.

Core Safety Rules:
  1. VIOLATION (Danger / Red):
     - At least one bare head detected without a helmet (head_count > 0).
     - Workers are in immediate danger of falling debris / head injuries.

  2. COMPLIANT (Safe / Green):
     - All visible workers are wearing safety helmets (head_count == 0, helmet_count > 0).
     - Site adheres to workplace safety regulations.

  3. UNKNOWN / CAUTION (Warning / Yellow):
     - Worker body detected, but head/helmet is occluded or outside frame.

  4. NO WORKERS (Neutral / Gray):
     - No personnel or safety gear detected in the image.
==============================================================================
"""

from typing import List, Dict, Any


def evaluate_compliance(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes a list of YOLO detections and determines the safety compliance status.

    Parameters:
        detections (list): List of detected objects, where each object is a dictionary:
                           {
                               "class_name": "helmet" | "head" | "person",
                               "confidence": float (0.0 to 1.0),
                               "box": [x1, y1, x2, y2]
                           }

    Returns:
        dict: Complete safety assessment report containing:
              - is_safe (bool): True if 100% compliant, False if violations exist
              - status (str): "COMPLIANT" | "VIOLATION" | "UNKNOWN" | "NO_WORKERS"
              - status_level (str): "safe" | "danger" | "warning" | "neutral"
              - message (str): Human-friendly description for supervisors
              - compliance_rate (float): Percentage of protected workers (0.0% - 100.0%)
              - counts (dict): Breakdown of helmets, unprotected heads, and workers
    """
    # 1. Count occurrences of each class from the detections
    helmet_count = sum(1 for d in detections if d.get("class_name") == "helmet")
    head_count = sum(1 for d in detections if d.get("class_name") == "head")
    person_count = sum(1 for d in detections if d.get("class_name") == "person")

    # Total workers evaluated based on heads visible
    total_heads_evaluated = helmet_count + head_count

    # 2. Calculate compliance percentage
    if total_heads_evaluated > 0:
        compliance_rate = round((helmet_count / total_heads_evaluated) * 100, 1)
    elif person_count > 0:
        # Worker visible, but headgear couldn't be confirmed
        compliance_rate = 0.0
    else:
        # No workers present in the frame
        compliance_rate = 100.0

    # 3. Apply Safety Business Logic
    if head_count > 0:
        # Critical violation: at least one worker without a hard hat
        return {
            "is_safe": False,
            "status": "VIOLATION",
            "status_level": "danger",
            "color_hex": "#EF4444",  # Red
            "message": f"SAFETY ALERT: {head_count} worker(s) detected without safety helmets!",
            "compliance_rate": compliance_rate,
            "counts": {
                "helmet": helmet_count,
                "head": head_count,
                "person": person_count,
                "total_heads_evaluated": total_heads_evaluated
            }
        }

    elif helmet_count > 0:
        # Safe: workers are present and all have helmets
        return {
            "is_safe": True,
            "status": "COMPLIANT",
            "status_level": "safe",
            "color_hex": "#10B981",  # Green
            "message": f"ALL CLEAR: All {helmet_count} worker(s) are wearing required safety helmets.",
            "compliance_rate": compliance_rate,
            "counts": {
                "helmet": helmet_count,
                "head": head_count,
                "person": person_count,
                "total_heads_evaluated": total_heads_evaluated
            }
        }

    elif person_count > 0:
        # Worker detected, but head/helmet not detected (e.g. back turned, occluded)
        return {
            "is_safe": False,
            "status": "UNKNOWN",
            "status_level": "warning",
            "color_hex": "#F59E0B",  # Orange/Amber
            "message": f"CAUTION: {person_count} worker(s) detected, but headgear is not clearly visible.",
            "compliance_rate": compliance_rate,
            "counts": {
                "helmet": helmet_count,
                "head": head_count,
                "person": person_count,
                "total_heads_evaluated": total_heads_evaluated
            }
        }

    else:
        # Empty scene: no workers or gear detected
        return {
            "is_safe": True,
            "status": "NO_WORKERS",
            "status_level": "neutral",
            "color_hex": "#6B7280",  # Gray
            "message": "NO PERSONNEL: No workers or safety gear detected in the image.",
            "compliance_rate": compliance_rate,
            "counts": {
                "helmet": 0,
                "head": 0,
                "person": 0,
                "total_heads_evaluated": 0
            }
        }


# Quick self-test demonstration if run directly
if __name__ == "__main__":
    print("Testing Compliance Rules Engine...")

    # Test Case 1: Safe situation (2 helmets)
    test_detections_safe = [
        {"class_name": "helmet", "confidence": 0.95, "box": [10, 10, 50, 50]},
        {"class_name": "helmet", "confidence": 0.91, "box": [100, 20, 140, 60]}
    ]
    report_safe = evaluate_compliance(test_detections_safe)
    print("\nCase 1 (Safe):", report_safe["status"], "-", report_safe["message"])

    # Test Case 2: Violation situation (1 helmet, 1 bare head)
    test_detections_violation = [
        {"class_name": "helmet", "confidence": 0.92, "box": [10, 10, 50, 50]},
        {"class_name": "head", "confidence": 0.88, "box": [100, 20, 140, 60]}
    ]
    report_violation = evaluate_compliance(test_detections_violation)
    print("Case 2 (Violation):", report_violation["status"], "-", report_violation["message"])
