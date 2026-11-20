"""
prepare_data.py
Industrial Surface Defect Inspection Platform (CSAI3027P)
Dataset audit, cryptographic hash duplication check, cross-split leakage audit,
VOC XML parsing, multi-box statistics, and reproducible 70/15/15 stratified train/val/test splitting.
"""

import os
import glob
import json
import csv
import random
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from PIL import Image

# Fixed random seed for total reproducibility
SEED = 42
random.seed(SEED)

def get_dataset_root():
    """Read dataset location from config/dataset_path.txt."""
    config_file = os.path.join("config", "dataset_path.txt")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, "r") as f:
        rel_path = f.read().strip()
    return os.path.abspath(rel_path)

def create_directory_structure():
    """Create all required project folders."""
    dirs = [
        "fundamentals",
        "models",
        "models/checkpoints",
        "advanced",
        "app",
        "app/static",
        "experiments",
        "outputs",
        "outputs/figures",
        "outputs/metrics",
        "docs",
        "report",
        "tests",
        "config",
        "data/splits",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[INIT] Verified repository directory structure.")

def compute_file_hashes(file_path):
    """Compute SHA-256 and MD5 cryptographic hashes of a file."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()

def parse_voc_xml(xml_path):
    """
    Parse a Pascal VOC XML annotation file.
    Returns: list of dicts with bbox and label metadata.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename_elem = root.find("filename")
    filename = filename_elem.text.strip() if filename_elem is not None else ""
    
    size_elem = root.find("size")
    width = int(size_elem.find("width").text) if size_elem is not None else 200
    height = int(size_elem.find("height").text) if size_elem is not None else 200
    depth = int(size_elem.find("depth").text) if size_elem is not None else 1
    
    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        # Clamp coordinates to image boundaries
        xmin = max(0.0, min(float(width), xmin))
        ymin = max(0.0, min(float(height), ymin))
        xmax = max(0.0, min(float(width), xmax))
        ymax = max(0.0, min(float(height), ymax))
        
        objects.append({
            "class": class_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            # Normalized coordinates [0, 1] for detector targets
            "norm_xmin": round(xmin / width, 4),
            "norm_ymin": round(ymin / height, 4),
            "norm_xmax": round(xmax / width, 4),
            "norm_ymax": round(ymax / height, 4),
        })
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "depth": depth,
        "objects": objects
    }

def audit_and_split():
    """
    Perform rigorous dataset audit, cryptographic hash duplication check,
    split leakage audit, and stratified 70/15/15 train/val/test split.
    """
    dataset_root = get_dataset_root()
    images_dir = os.path.join(dataset_root, "IMAGES")
    annotations_dir = os.path.join(dataset_root, "ANNOTATIONS")
    
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    xml_paths = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))
    
    print(f"[AUDIT] Dataset root: {dataset_root}")
    print(f"[AUDIT] Images found: {len(image_paths)}")
    print(f"[AUDIT] XMLs found:   {len(xml_paths)}")
    
    expected_classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(expected_classes)}
    
    # Hash auditing structures
    sha256_map = defaultdict(list)
    md5_map = defaultdict(list)
    
    class_groups = defaultdict(list)
    manifest_records = []
    
    box_counts_per_class = Counter()
    multi_box_dist = Counter()
    
    print("\n[AUDIT] Computing cryptographic hashes and verifying image-annotation pairs...")
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        stem, _ = os.path.splitext(base_name)
        primary_class = stem.rsplit("_", 1)[0]
        
        if primary_class not in expected_classes:
            raise ValueError(f"Unexpected class '{primary_class}' for image: {img_path}")
        
        xml_path = os.path.join(annotations_dir, f"{stem}.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Missing annotation for image: {img_path}")
        
        # Compute hashes
        img_sha256, img_md5 = compute_file_hashes(img_path)
        sha256_map[img_sha256].append(base_name)
        md5_map[img_md5].append(base_name)
        
        # Verify XML
        xml_info = parse_voc_xml(xml_path)
        
        # Verify image integrity & dimensions
        with Image.open(img_path) as im:
            im_w, im_h = im.size
            im_mode = im.mode
            
        num_boxes = len(xml_info["objects"])
        multi_box_dist[num_boxes] += 1
        for obj in xml_info["objects"]:
            box_counts_per_class[obj["class"]] += 1
        
        record = {
            "image_id": stem,
            "filename": base_name,
            "primary_class": primary_class,
            "class_id": class_to_id[primary_class],
            "image_path": os.path.relpath(img_path, os.getcwd()).replace("\\", "/"),
            "xml_path": os.path.relpath(xml_path, os.getcwd()).replace("\\", "/"),
            "sha256": img_sha256,
            "md5": img_md5,
            "width": im_w,
            "height": im_h,
            "mode": im_mode,
            "num_boxes": num_boxes,
            "boxes": xml_info["objects"]
        }
        class_groups[primary_class].append(record)
        manifest_records.append(record)
        
    # Check for duplicate hashes in raw data
    duplicate_sha256 = {h: files for h, files in sha256_map.items() if len(files) > 1}
    duplicate_md5 = {h: files for h, files in md5_map.items() if len(files) > 1}
    
    print("\n[AUDIT: DUPLICATE HASH CHECK]")
    print(f"  - Total unique SHA-256 hashes: {len(sha256_map)} / {len(image_paths)}")
    print(f"  - Total unique MD5 hashes:    {len(md5_map)} / {len(image_paths)}")
    if duplicate_sha256:
        print(f"  - Found {len(duplicate_sha256)} hash collision cluster(s):")
        for h, files in duplicate_sha256.items():
            print(f"    Hash: {h} -> {files}")
    else:
        print("  - [PASS] Zero duplicate images found: All 1,800 images have 100% unique cryptographic hashes.")

    print("\n[AUDIT: CLASS & INSTANCE COUNTS]")
    for cls_name in expected_classes:
        n_img = len(class_groups[cls_name])
        n_box = box_counts_per_class[cls_name]
        mean_box = n_box / n_img if n_img else 0
        print(f"  - {cls_name:16s}: {n_img:3d} images, {n_box:4d} boxes ({mean_box:.2f} boxes/img)")
    total_boxes = sum(box_counts_per_class.values())
    print(f"  - TOTAL:           {len(image_paths):4d} images, {total_boxes:4d} boxes ({total_boxes/len(image_paths):.2f} boxes/img)")

    print("\n[AUDIT: MULTI-BOX DISTRIBUTION]")
    for n_b in sorted(multi_box_dist.keys()):
        pct = (multi_box_dist[n_b] / len(image_paths)) * 100
        print(f"  - {n_b:2d} box(es): {multi_box_dist[n_b]:4d} images ({pct:5.2f}%)")

    # Stratified 70% Train (210/class), 15% Val (45/class), 15% Test (45/class)
    # Guaranteed Zero-Leakage: Any duplicate hashes must be grouped into the SAME split (Train).
    train_records, val_records, test_records = [], [], []
    
    for cls_name in expected_classes:
        records = class_groups[cls_name]
        
        # Group records by hash to prevent duplicate separation across splits
        hash_to_records = defaultdict(list)
        for r in records:
            hash_to_records[r["sha256"]].append(r)
            
        # Deterministic shuffle of unique hash clusters
        rng = random.Random(SEED + class_to_id[cls_name])
        
        # Put duplicate clusters directly into train to avoid leakage into val/test
        train_cluster = []
        regular_clusters = []
        for h, cluster in hash_to_records.items():
            if len(cluster) > 1:
                train_cluster.extend(cluster)
            else:
                regular_clusters.append(cluster[0])
                
        rng.shuffle(regular_clusters)
        
        # Target sizes: 210 train, 45 val, 45 test
        n_train_needed = 210 - len(train_cluster)
        c_train = train_cluster + regular_clusters[:n_train_needed]
        c_val = regular_clusters[n_train_needed:n_train_needed + 45]
        c_test = regular_clusters[n_train_needed + 45:]
        
        assert len(c_train) == 210, f"Expected 210 train samples for {cls_name}, got {len(c_train)}"
        assert len(c_val) == 45, f"Expected 45 val samples for {cls_name}, got {len(c_val)}"
        assert len(c_test) == 45, f"Expected 45 test samples for {cls_name}, got {len(c_test)}"
        
        for r in c_train:
            r["split"] = "train"
        for r in c_val:
            r["split"] = "val"
        for r in c_test:
            r["split"] = "test"
            
        train_records.extend(c_train)
        val_records.extend(c_val)
        test_records.extend(c_test)
        
    print(f"\n[SPLIT] Partitioning complete (Seed {SEED}):")
    print(f"  - Train: {len(train_records)} images (70.0%) - {len(train_records)//6} per class")
    print(f"  - Val:   {len(val_records)} images (15.0%) - {len(val_records)//6} per class")
    print(f"  - Test:  {len(test_records)} images (15.0%) - {len(test_records)//6} per class")
    print(f"  - Total: {len(train_records) + len(val_records) + len(test_records)} images")
    
    # Rigorous Cross-Split Leakage Audit
    train_hashes = set(r["sha256"] for r in train_records)
    val_hashes = set(r["sha256"] for r in val_records)
    test_hashes = set(r["sha256"] for r in test_records)
    
    train_val_leak = train_hashes.intersection(val_hashes)
    train_test_leak = train_hashes.intersection(test_hashes)
    val_test_leak = val_hashes.intersection(test_hashes)
    
    print("\n[AUDIT: CROSS-SPLIT LEAKAGE VERIFICATION]")
    print(f"  - Train vs Val hash collisions:  {len(train_val_leak)}")
    print(f"  - Train vs Test hash collisions: {len(train_test_leak)}")
    print(f"  - Val vs Test hash collisions:   {len(val_test_leak)}")
    if not (train_val_leak or train_test_leak or val_test_leak):
        print("  - [PASS] Zero cross-split data leakage verified: 100% split isolation.")
    else:
        raise RuntimeError(f"CRITICAL ERROR: Cross-split data leakage detected! Collisions: {train_val_leak | train_test_leak | val_test_leak}")

    all_split_records = train_records + val_records + test_records
    
    # Save Split Manifest (JSON)
    json_path = os.path.join("data", "splits", "split_manifest.json")
    with open(json_path, "w") as f:
        json.dump({
            "seed": SEED,
            "classes": expected_classes,
            "class_to_id": class_to_id,
            "counts": {
                "train": len(train_records),
                "val": len(val_records),
                "test": len(test_records),
                "total": len(all_split_records)
            },
            "records": all_split_records
        }, f, indent=2)
    print(f"[SAVE] Split manifest JSON: {json_path}")
    
    # Save Flat CSV for detection & tabular analysis
    csv_path = os.path.join("data", "splits", "split_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filename", "primary_class", "class_id", "split", "sha256", "image_path", "xml_path", "width", "height", "num_boxes", "boxes_json"])
        for r in all_split_records:
            writer.writerow([
                r["image_id"],
                r["filename"],
                r["primary_class"],
                r["class_id"],
                r["split"],
                r["sha256"],
                r["image_path"],
                r["xml_path"],
                r["width"],
                r["height"],
                r["num_boxes"],
                json.dumps(r["boxes"])
            ])
    print(f"[SAVE] Split manifest CSV:  {csv_path}")

    # Save Hash Audit JSON report
    hash_audit_path = os.path.join("data", "splits", "hash_audit.json")
    hash_audit_summary = {
        "dataset_name": "NEU-DET",
        "total_images": len(image_paths),
        "total_unique_sha256": len(sha256_map),
        "total_unique_md5": len(md5_map),
        "duplicate_images_count": sum(len(v) for v in duplicate_sha256.values()),
        "duplicate_hash_clusters": len(duplicate_sha256),
        "duplicate_sets": duplicate_sha256,
        "splits": {
            "train_count": len(train_records),
            "val_count": len(val_records),
            "test_count": len(test_records),
            "train_unique_hashes": len(train_hashes),
            "val_unique_hashes": len(val_hashes),
            "test_unique_hashes": len(test_hashes),
        },
        "leakage_audit": {
            "train_val_leak_count": len(train_val_leak),
            "train_test_leak_count": len(train_test_leak),
            "val_test_leak_count": len(val_test_leak),
            "is_leak_free": bool(not (train_val_leak or train_test_leak or val_test_leak))
        },
        "multi_box_distribution": {str(k): v for k, v in sorted(multi_box_dist.items())},
        "boxes_per_class": dict(box_counts_per_class),
        "total_boxes": total_boxes
    }
    with open(hash_audit_path, "w") as f:
        json.dump(hash_audit_summary, f, indent=2)
    print(f"[SAVE] Hash & Leakage audit JSON: {hash_audit_path}")

    # Copy 6 sample test images (one per class) to app/static/samples/ for the demo gallery
    samples_dir = os.path.join("app", "static", "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sample_registry = []
    
    for cls_name in expected_classes:
        # Pick the first test sample for each class
        sample = next(r for r in test_records if r["primary_class"] == cls_name)
        src_img = os.path.abspath(sample["image_path"])
        dst_img = os.path.join(samples_dir, sample["filename"])
        
        # Read and save clean copy
        with Image.open(src_img) as im:
            im.save(dst_img)
            
        sample_registry.append({
            "class_name": cls_name,
            "filename": sample["filename"],
            "url": f"/static/samples/{sample['filename']}",
            "sample_boxes": sample["boxes"]
        })
        
    sample_reg_path = os.path.join("app", "static", "samples", "sample_registry.json")
    with open(sample_reg_path, "w") as f:
        json.dump(sample_registry, f, indent=2)
    print(f"[SAVE] Prepared sample gallery: {sample_reg_path} ({len(sample_registry)} samples)")

if __name__ == "__main__":
    create_directory_structure()
    audit_and_split()
