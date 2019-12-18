import cv2
from yolo.yolo_opencv import ObjectDetection
from deepsort import *
import sys
from threading import Thread
import time
class ObjectTracking:
    def __init__(self):
        self.object_detection = ObjectDetection()
        self.deepsort = deepsort_rbc()
        self.result = []
    def tracking(self,cap):
        frame_id = 1
        objects =[]
        while True:
            print(frame_id)

            ret, frame = cap.read()
            frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
            if ret is False:
                frame_id += 1
                break

            if frame_id%1==0:
                objects = self.object_detection.detect(frame)
            detections = []
            out_scores = []
            labels = []
            for obj in objects:
                detections.append([obj['x'], obj['y'], obj['w'], obj['h']])
                out_scores.append(obj['confidences'])
                labels.append(obj['label'])
            if len(detections) == 0:
                print("No dets")
                frame_id += 1
                continue

            detections = np.array(detections)
            out_scores = np.array(out_scores)

            tracker, detections_class = self.deepsort.run_deep_sort(frame, out_scores, detections, labels)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                id_num = str(track.track_id)  # Get the ID for the particular track.
                features = track.features  # Get the feature vector corresponding to the detection.

                # Draw bbox from tracker.
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 1)
                cv2.putText(frame, str(id_num), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 150, (0, 255, 0), 2)

                # Draw bbox from detector. Just to compare.
                for det, label in detections_class:
                    bbox = det.to_tlbr()
                    if label != None:
                        cv2.putText(frame, label, (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 150, (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 1)
            frame_id += 1
            self.result.append(frame)

if __name__ == '__main__':

    path_in = 'input.avi'  # Đường dẫn đến file video
    path_out = 'output.avi'  # Đường dẫn lưu file video
    if len(sys.argv) >= 2:
        path_in = sys.argv[1]
    if len(sys.argv) >= 3:
        path_out = sys.argv[2]
    object_tracking = ObjectTracking()
    cap = cv2.VideoCapture(path_in)
    th = Thread(target=object_tracking.tracking,args=[cap])
    th.start()
    time.sleep(5)
    while True:
        if len(object_tracking.result)>0:
            cv2.imshow('frame', object_tracking.result.pop(0))
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


