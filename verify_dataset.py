import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from collections import Counter

# Set up paths
workspace_root = os.path.abspath(".")
dataset_path = os.path.join(workspace_root, "data", "raw", "NEU-DET")
images_dir = os.path.join(dataset_path, "IMAGES")
annotations_dir = os.path.join(dataset_path, "ANNOTATIONS")

print(f"=== Dataset Verification ===")
print(f"Dataset Path: {dataset_path}")

# Check directories exist
assert os.path.isdir(images_dir), f"IMAGES directory not found at {images_dir}"
assert os.path.isdir(annotations_dir), f"ANNOTATIONS directory not found at {annotations_dir}"

# Find all image and annotation files
image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
annotation_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))

print(f"Total .jpg images found: {len(image_files)}")
print(f"Total .xml annotations found: {len(annotation_files)}")

# Count images per class based on filename prefix (e.g. crazing_1.jpg)
img_class_counts = Counter()
for img_path in image_files:
    fname = os.path.basename(img_path)
    cls_name = fname.rsplit('_', 1)[0]
    img_class_counts[cls_name] += 1

print("\n--- Image Count per Class (from filenames) ---")
for cls, count in sorted(img_class_counts.items()):
    print(f"  {cls}: {count}")

# Verify against the expected 6 classes
expected_classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
found_classes = sorted(list(img_class_counts.keys()))
print(f"\nExpected classes: {expected_classes}")
print(f"Found classes:    {found_classes}")
print(f"All expected classes match: {found_classes == sorted(expected_classes)}")

# Sample image size
sample_img_path = image_files[0]
with Image.open(sample_img_path) as img:
    sample_size = img.size  # (width, height)
    sample_mode = img.mode
    sample_format = img.format
print(f"\n--- Sample Image Details ---")
print(f"File: {os.path.basename(sample_img_path)}")
print(f"Dimensions: {sample_size[0]} x {sample_size[1]} (width x height)")
print(f"Color mode: {sample_mode}")
print(f"Format: {sample_format}")

# Sample XML parsed
sample_xml_path = annotation_files[0]
tree = ET.parse(sample_xml_path)
root = tree.getroot()

print(f"\n--- Sample XML Parsed ---")
print(f"File: {os.path.basename(sample_xml_path)}")
filename_elem = root.find("filename")
xml_filename = filename_elem.text if filename_elem is not None else "N/A"
print(f"Filename in XML: {xml_filename}")

size_elem = root.find("size")
if size_elem is not None:
    width = size_elem.find("width").text
    height = size_elem.find("height").text
    depth = size_elem.find("depth").text
    print(f"Image Size in XML: {width} x {height} x {depth}")

objects = root.findall("object")
print(f"Total Objects in sample XML: {len(objects)}")
for idx, obj in enumerate(objects, 1):
    name = obj.find("name").text
    bndbox = obj.find("bndbox")
    xmin = bndbox.find("xmin").text
    ymin = bndbox.find("ymin").text
    xmax = bndbox.find("xmax").text
    ymax = bndbox.find("ymax").text
    print(f"  Object {idx}: Class='{name}', BoundingBox=[xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}]")

# Count annotations / object occurrences across all XMLs
xml_class_counts = Counter()
for xml_file in annotation_files:
    t = ET.parse(xml_file)
    r = t.getroot()
    for o in r.findall("object"):
        n = o.find("name").text
        xml_class_counts[n] += 1

print("\n--- Object Instance Count per Class (from XML annotations) ---")
for cls, count in sorted(xml_class_counts.items()):
    print(f"  {cls}: {count} bounding boxes")

# Save confirmed path to config/dataset_path.txt
config_dir = os.path.join(workspace_root, "config")
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, "dataset_path.txt")
relative_dataset_path = "data/raw/NEU-DET"

with open(config_file, "w") as f:
    f.write(relative_dataset_path + "\n")

print(f"\nSaved dataset path to {config_file}:")
with open(config_file, "r") as f:
    print(f"Content: '{f.read().strip()}'")
