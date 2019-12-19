import cv2
import numpy as np
import os

root = os.path.dirname(__file__)


class ObjectDetection:
    def __init__(self, path_model=os.path.join(root, 'yolo3.weights'),
                 path_config=os.path.join(root, 'yolo3.cfg'),
                 path_labels=os.path.join(root, 'yolo3.txt')):
        with open(path_labels, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = cv2.dnn.readNet(path_model, path_config)
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        print(self.ln)

    def detect(self, image):
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (320, 320), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)

        # 'yolo_82', 'yolo_94', 'yolo_106'
        outs = self.net.forward(self.ln)
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        result = []
        for i in indices:
            i = i[0]

            box = boxes[i]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            result.append({
                'label': self.classes[class_ids[i]],
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'confidences': confidences[i]
            })

        return result
