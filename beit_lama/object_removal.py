from transformers import BeitImageProcessor, BeitForSemanticSegmentation
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

class ObjectRemoval(object):
    def __init__(self, **data):
        self.input_path = data.get("input")
        self.image = Image.open(self.input_path)

    def generate_segment(self):
        model_name = "microsoft/beit-base-finetuned-ade-640-640"
        feature_extractor = BeitImageProcessor(do_resize=True, size=640, do_center_crop=False)
        model = BeitForSemanticSegmentation.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        pixel_values = feature_extractor(self.image, return_tensors="pt").pixel_values.to(device)
        outputs = model(pixel_values)

        logits = nn.functional.interpolate(outputs.logits,
                size=self.image.size[::-1],
                mode='bilinear',
                align_corners=False)

        seg = logits.argmax(dim=1)[0].cpu()
        self.color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

        return self.color_seg

    def choose_segment(self, **data):
        x = data.get("x")
        y = data.get("y")
        output_path = data.get("output")

        mask_seg = np.array(self.color_seg, copy=True)

        h, w = mask_seg.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(mask_seg, mask, (1150, 875), -1);

        mask_seg_clone = np.array(mask_seg, copy=True)

        mask_color = np.array([0, 0, 0])

        for i, v in enumerate(mask_seg):
            for j, v1 in enumerate(v):
                if (v1 == mask_color).all():
                    mask_seg_clone[i][j] = [255, 255, 255]
                else:
                    mask_seg_clone[i][j] = [0, 0, 0]

        cv2.imwrite(output_path, mask_seg_clone)

        return mask_seg_clone
