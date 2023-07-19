import torch, onnx
import onnxruntime as ort
import cv2
import numpy as np
import io
from PIL import Image
import time
import torchvision
import os
from tqdm import tqdm

from character_recognition import CharacterRecognizer
recognizer = CharacterRecognizer(verbose=False)


def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
        new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, r, (left, top)

def process_image(img_src, img_size, stride, half):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size, stride=stride)[0]
    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0
    return image, img_src

# with open(test_path, 'rb') as f:
#     img_np = np.asarray(Image.open(io.BytesIO(f.read())))
def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, multi_label=False):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_det = 50 # maximum detection objects
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output

def button_candidates(boxes, scores, image):

    button_scores = []  # stores the score of each button (confidence)
    button_patches = []  # stores the cropped image that encloses the button
    button_positions = []  # stores the coordinates of the bounding box on buttons

    for box, score in zip(boxes, scores):
        if score < 0.5:
            continue

        y_min = int(box[0])
        x_min = int(box[1])
        y_max = int(box[2])
        x_max = int(box[3])

        if x_min < 0 or y_min < 0:
            continue
        button_patch = image[y_min: y_max, x_min: x_max]
        button_patch = cv2.resize(button_patch, (180, 180))

        button_scores.append(score)
        button_patches.append(button_patch)
        button_positions.append([x_min, y_min, x_max, y_max])
    return button_patches, button_positions, button_scores
st = time.time()

test_paths = []
for file_name in os.listdir("./test_imgs/"):
    test_paths.append("./test_imgs/"+file_name)

for test_path in tqdm(test_paths):
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
    # dets = np.array(dets)
    # print(dets)
    dets = dets.tolist()
    boxes = [row[:4] for row in dets]
    scores = [row[4] for row in dets]

    button_patches, button_positions, _ = button_candidates(boxes, scores, img_np)

    for button_img in button_patches:
        # get button text and button_score for each of the images in button_patches
        button_text, button_score, _ = recognizer.predict(button_img)
        # print(button_text)

print(time.time()-st)