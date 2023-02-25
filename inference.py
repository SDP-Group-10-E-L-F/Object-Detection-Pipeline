import torch
import os
import glob
import gc

torch.cuda.empty_cache()
gc.collect()

# Model
model = torch.hub.load('.', 'custom', path='best.pt', source='local')

# Images
dir = "/home/zheng/Documents/mlops/YOLO/YOLOv7/datasets/fsoco_v2_yolov7_labels/test/images/"
imgs = glob.glob(f"{dir}*.jpg")
# imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batch of images

# Inference
results = model(imgs)
results.print()  # or .show(), .save()
# Speed: 631.5ms pre-process, 19.2ms inference, 1.6ms NMS per image at shape (2, 3, 640, 640)