#Inference on a test image from yolov6 onnx and OCR
import torch, onnx
import onnxruntime as ort
import cv2
import numpy as np
from util import process_image, non_max_suppression,button_candidates
from character_recognition import CharacterRecognizer

#load model
onnx_model = onnx.load('models/yolov6/best_ckpt.onnx')
onnx.checker.check_model(onnx_model)
ort_sess = ort.InferenceSession('models/yolov6/best_ckpt.onnx')
recognizer = CharacterRecognizer(verbose=False)

#load image
test_path = "test_imgs/15_jpg.rf.7e4ba4c0c0bdb3beea120118c56fd793.jpg"
img_np = cv2.imread(test_path)
img_n,_ = process_image(img_np,(416,416),2,False)
img_n = img_n.numpy()[np.newaxis]

#run pipeline
preds = ort_sess.run(None, {'images': img_n})
preds = torch.tensor((np.array(preds)).reshape(1,3549,6))
dets = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, agnostic=False)[0].tolist()
boxes, scores = [row[:4] for row in dets], [row[4] for row in dets]

button_patches, button_positions, _ = button_candidates(boxes, scores, img_np)
for button_img, button_pos in zip(button_patches, button_positions):
        button_text, button_score, button_draw =recognizer.predict(button_img, draw=True)
        x_min, y_min, x_max, y_max = button_pos
        button_rec = cv2.resize(button_draw, (x_max-x_min, y_max-y_min))
        img_np[y_min+6:y_max-6, x_min+6:x_max-6] = button_rec[6:-6, 6:-6]

#view results
cv2.imshow('Image',img_np)
cv2.waitKey(0)
