"""
test_retinanet_train.py
Benchmark step latency and compatibility of Keras-CV RetinaNet on CPU with Keras 3.
"""

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from PIL import Image

import keras_cv
from keras_cv.models import RetinaNet, ResNet50Backbone

print(f"[TEST] TensorFlow {tf.__version__}, Keras-CV {keras_cv.__version__}")

class PatchedRetinaNet(RetinaNet):
    def train_step(self, *args):
        data = args[-1]
        if isinstance(data, (tuple, list)):
            if isinstance(data[0], dict) and "images" in data[0] and "bounding_boxes" in data[0]:
                x = data[0]["images"]
                y = data[0]["bounding_boxes"]
            elif len(data) >= 2:
                x = data[0]
                y = data[1]
            else:
                x, y = data[0], None
        elif isinstance(data, dict):
            x = data["images"]
            y = data["bounding_boxes"]
        else:
            x, y = data

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=predictions, sample_weight=None)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

with open("data/splits/split_manifest.json", "r") as f:
    manifest = json.load(f)

records = manifest["records"][:8]
classes = manifest["classes"]
class_to_id = manifest["class_to_id"]

images = []
boxes_list = []
classes_list = []

for r in records:
    with Image.open(r["image_path"]) as im:
        im_resized = im.resize((224, 224))
        im_arr = np.array(im_resized.convert("RGB"), dtype=np.float32) / 255.0
    images.append(im_arr)
    
    # Scale boxes to 224x224
    b_coords = []
    b_classes = []
    for b in r["boxes"]:
        # b is on 200x200; scale to 224x224
        scale = 224.0 / 200.0
        b_coords.append([b["xmin"] * scale, b["ymin"] * scale, b["xmax"] * scale, b["ymax"] * scale])
        b_classes.append(class_to_id[b["class"]])
        
    boxes_list.append(b_coords)
    classes_list.append(b_classes)

images_tensor = tf.constant(images, dtype=tf.float32)
boxes_ragged = tf.ragged.constant(boxes_list, dtype=tf.float32)
classes_ragged = tf.ragged.constant(classes_list, dtype=tf.float32)

print("[TEST] Building PatchedRetinaNet with ResNet50 backbone...")
backbone = ResNet50Backbone()
model = PatchedRetinaNet(
    num_classes=len(classes),
    bounding_box_format="xyxy",
    backbone=backbone
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    classification_loss="focal",
    box_loss="SmoothL1"
)

packed_input = {
    "images": images_tensor[:2],
    "bounding_boxes": {
        "boxes": boxes_ragged[:2],
        "classes": classes_ragged[:2]
    }
}

print("[TEST] Running 1 training step on CPU for batch size 2...")
t0 = time.time()
try:
    loss_dict = model.train_on_batch(x=packed_input)
    step_time = time.time() - t0
    print(f"[BENCHMARK] Step time for batch size 2 on CPU: {step_time:.2f} seconds")
    print(f"[BENCHMARK] Loss dict: {loss_dict}")
    estimated_epoch_time = (1260 / 2) * step_time / 60.0
    print(f"[BENCHMARK] Estimated 1 Epoch time on CPU (1260 images): {estimated_epoch_time:.1f} minutes")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[ERROR] PatchedRetinaNet train_on_batch encountered an error: {e}")
