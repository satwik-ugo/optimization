import torch, onnx
import onnxruntime as ort
import cv2
import numpy as np
import time
import os
from tqdm import tqdm
from util import process_image, non_max_suppression, button_candidates
from character_recognition import CharacterRecognizer
recognizer = CharacterRecognizer(verbose=False)

test_paths = []
for file_name in os.listdir("./test_imgs/"):
    test_paths.append("./test_imgs/"+file_name)

st = time.time()
for test_path in tqdm(test_paths[0:100]):
    img_np = cv2.imread(test_path)
    img_n,_ = process_image(img_np,(416,416),32,False)
    img_n = img_n.numpy()[np.newaxis]

    #load model
    onnx_model = onnx.load('models/yolov6/best_ckpt.onnx')
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession('models/yolov6/best_ckpt.onnx')
    preds = ort_sess.run(None, {'images': img_n})
    preds = np.array(preds)
    preds = preds.reshape(1,3549,6)
    preds = torch.tensor(preds)
    dets = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, agnostic=False)[0]

    dets = dets.tolist()
    boxes = [row[:4] for row in dets]
    scores = [row[4] for row in dets]

    button_patches, button_positions, _ = button_candidates(boxes, scores, img_np)

    for button_img in button_patches:
        # get button text and button_score for each of the images in button_patches
        button_text, button_score, _ = recognizer.predict(button_img)

print(f"Time for 100 images (yolov6onnx + OCR) is {time.time()-st}")