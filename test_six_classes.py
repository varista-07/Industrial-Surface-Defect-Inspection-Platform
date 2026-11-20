import os
import requests
import json

BASE_URL = "http://127.0.0.1:8000"
IMAGES_DIR = os.path.join("data", "raw", "NEU-DET", "IMAGES")

classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
sample_files = {
    "crazing": "crazing_1.jpg",
    "inclusion": "inclusion_1.jpg",
    "patches": "patches_1.jpg",
    "pitted_surface": "pitted_surface_1.jpg",
    "rolled-in_scale": "rolled-in_scale_1.jpg",
    "scratches": "scratches_1.jpg"
}

results = []
print("=" * 80)
print("TESTING ALL 6 NEU-DET DEFECT CLASSES VIA LIVE FASTAPI ENDPOINT")
print("=" * 80)

for cls_name in classes:
    fname = sample_files[cls_name]
    fpath = os.path.join(IMAGES_DIR, fname)
    assert os.path.exists(fpath), f"File not found: {fpath}"
    
    with open(fpath, "rb") as f:
        file_bytes = f.read()
        
    resp = requests.post(
        f"{BASE_URL}/predict?threshold=0.70",
        files={"file": (fname, file_bytes, "image/jpeg")}
    )
    
    assert resp.status_code == 200, f"Error {resp.status_code}: {resp.text}"
    data = resp.json()
    
    pred_cls = data["predicted_class"]
    conf = data["confidence_percentage"]
    verdict = data["verdict"]
    bbox = data["bounding_box"]
    report = data["inspection_report"]
    latency = data["inference_latency_ms"]
    
    viz = data["visualizations"]
    has_orig = viz.get("original_image") is not None
    has_bbox = viz.get("bounding_box_overlay") is not None
    has_gradcam = viz.get("gradcam_heatmap_overlay") is not None
    has_unet = viz.get("segmentation_mask_overlay") is not None
    
    print(f"\n[CLASS: {cls_name.upper()}] (File: {fname})")
    print(f"  Predicted Class:       {pred_cls} ({'MATCH' if pred_cls == cls_name else 'MISMATCH'})")
    print(f"  Confidence:            {conf}%")
    print(f"  Verdict:               {verdict}")
    print(f"  Bounding Box [x1,y1,x2,y2]: {bbox}")
    print(f"  Visualizations:        Orig={has_orig}, BBox={has_bbox}, GradCAM={has_gradcam}, UNet={has_unet}")
    print(f"  Inference Latency:     {latency} ms")
    print(f"  Inspection Report:     {report}")
    
    results.append({
        "ground_truth": cls_name,
        "filename": fname,
        "predicted_class": pred_cls,
        "confidence_percentage": conf,
        "match": pred_cls == cls_name,
        "bounding_box": bbox,
        "latency_ms": latency,
        "has_gradcam": has_gradcam,
        "has_unet": has_unet,
        "report": report
    })

print("\n" + "=" * 80)
all_matched = all(r["match"] for r in results)
print(f"ALL 6 DEFECT CLASSES VERIFIED: {'SUCCESS (6/6 Correct)' if all_matched else 'SOME MISMATCH'}")
print("=" * 80)

with open("outputs/metrics/six_classes_verification.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved detailed verification results to outputs/metrics/six_classes_verification.json")
