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

onnx_model = onnx.load('models/yolov6/best_ckpt.onnx')
onnx.checker.check_model(onnx_model)
ort_sess = ort.InferenceSession('models/yolov6/best_ckpt.onnx')

test_paths = []
for file_name in os.listdir("./test_imgs/"):
    test_paths.append("./test_imgs/"+file_name)

st = time.time()
for test_path in tqdm(test_paths[0:100]):
    img_np = cv2.imread(test_path)
    img_n,_ = process_image(img_np,(416,416),32,False)
    img_n = img_n.numpy()[np.newaxis]
    # print(test_path)
    preds = ort_sess.run(None, {'images': img_n})
    preds = np.array(preds)
    preds = preds.reshape(1,3549,6)
    preds = torch.tensor(preds)
    dets = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, agnostic=False)[0]

    dets = dets.tolist()
    boxes = [row[:4] for row in dets]
    scores = [row[4] for row in dets]

    button_patches, button_positions, _ = button_candidates(boxes, scores, img_np)

    for button_img, button_pos in zip(button_patches, button_positions):
        button_text, button_score, button_draw =recognizer.predict(button_img, draw=True)
        #uncomment to visualize prediction results, note that times will not be relevant if you decide to visualize results in this script
        # x_min, y_min, x_max, y_max = button_pos
        # button_rec = cv2.resize(button_draw, (x_max-x_min, y_max-y_min))
        # try:
        #     img_np[y_min+6:y_max-6, x_min+6:x_max-6] = button_rec[6:-6, 6:-6]
        # except:
        #     continue
    
    # cv2.imshow('Image',img_np)
    # cv2.waitKey(0)

print(f"Time for 100 images (yolov6onnx + OCR) is {time.time()-st}") #comment this out if you want to visualize results