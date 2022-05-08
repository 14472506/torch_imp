import torchvision
import torchvision.transforms as T
import torch

from models import MaskRCNN_model, MaskRCNN_mobilenetv2

import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def get_coloured_mask(mask):
  """
  random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
  """
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(mask).astype(np.uint8)
  g = np.zeros_like(mask).astype(np.uint8)
  b = np.zeros_like(mask).astype(np.uint8)
  r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask




def get_prediction(img_path, confidence, COCO_CLASS_NAMES, model):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
      - confidence - threshold to keep the prediction or not
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
        ie: eg. segment of cat is made 1 and rest of the image is made 0
    
  """
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  # print(pred[0]['labels'].numpy().max())
  pred_class = [COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return masks, pred_boxes, pred_class



def segment_instance(img_path, COCO_CLASS_NAMES, model, confidence=0.5, rect_th=2, text_size=2, text_th=2):
  """
  segment_instance
    parameters:
      - img_path - path to input image
      - confidence- confidence to keep the prediction or not
      - rect_th - rect thickness
      - text_size
      - text_th - text thickness
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - each mask is added to the image in the ration 1:0.8 with opencv
      - final output is displayed
  """
  masks, boxes, pred_cls = get_prediction(img_path, confidence, COCO_CLASS_NAMES, model)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for i in range(len(masks)):
    rgb_mask = get_coloured_mask(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    b1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
    b2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
    cv2.rectangle(img, b1, b2, color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img,pred_cls[i], b1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
  plt.figure(figsize=(20,30))
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()
  


def main(model_path, num_classes, img_path):

    # load model: TODO, put this somewhere else i.e in models
    model = MaskRCNN_model(num_classes) 
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])

    # set model to eval mode
    model.eval()

    # define category names
    Labels = ['__background__', 'jersey_royal']

    segment_instance(img_path,
                     Labels,
                     model,
                     confidence=0.5,
                     rect_th=2,
                     text_size=2,
                     text_th=2
                     )



if __name__ == "__main__":
    
    model_path = "output/Mask_RCNN_R50_test/checkpoints/best_val_model.pth"
    num_classes = 2
    img_path = "data/jersey_royal_ds/val/118.JPG"

    main(model_path, num_classes, img_path)