import cv2
from yolo.yolo_opencv import ObjectDetection
from deepsort import *
import sys
from threading import Thread
import time
from collections import defaultdict


class ObjectTracking:
    def __init__(self, y0=100):
        self.object_detection = ObjectDetection(path_model='yolo/yolo3.weights',
                                                path_config='yolo/yolo3.cfg')
        self.deepsort = deepsort_rbc()
        self.objects = defaultdict(list)
        self.objects_direction = defaultdict(int)
        self.y = y0
        self.up = 0
        self.down = 0
        self.result = []
        self.is_running = True

    def stop(self):
        self.is_running = False

    def tracking(self, cap):
        frame_id = 1
        objects = []
        while self.is_running:
            print(frame_id)

            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (int(frame.shape[1] / 1.5), int(frame.shape[0] / 1.5)))
            if ret is False:
                frame_id += 1
                break

            if frame_id % 1 == 0:
                objects = self.object_detection.detect(frame)
            detections = []
            out_scores = []
            labels = []
            for obj in objects:
                # if obj['label'] != 'person':
                #     continue
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
            self.y = int(frame.shape[0] / 2)
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                id_num = str(track.track_id)  # Get the ID for the particular track.
                features = track.features  # Get the feature vector corresponding to the detection.
                self.objects[id_num].append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
                if len(self.objects[id_num]) >= 2:
                    l_obj = len(self.objects[id_num])
                    start_pos = self.objects[id_num][max(l_obj - 4, 0)]
                    curr_pos = self.objects[id_num][-1]
                    if (start_pos[1] - self.y) * (curr_pos[1] - self.y) < 0:
                        if curr_pos[1] > self.y:
                            if self.objects_direction[id_num] != -1:
                                self.down += 1
                                self.objects_direction[id_num] = -1
                        else:
                            if self.objects_direction[id_num] != 1:
                                self.up += 1
                                self.objects_direction[id_num] = 1
                # Draw bbox from tracker.
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 1)
                cv2.drawMarker(frame, (int(bbox[0] + bbox[2]) // 2, int(bbox[1] + bbox[3]) // 2), (0, 0, 255),
                               markerSize=4)
                cv2.putText(frame, str(id_num), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 150, (0, 255, 0), 2)
            cv2.line(frame, (0, self.y), (frame.shape[1], self.y), color=(0, 0, 255), thickness=1)
            cv2.putText(frame, 'up: {0}  down: {1}'.format(self.up, self.down), (4, int(frame.shape[0] * 0.9)), 0,
                        5e-3 * 150, (0, 255, 0), 2)
            # Draw bbox from detector. Just to compare.
            # for det, label in detections_class:
            #     bbox = det.to_tlbr()
            #     if label != None:
            #         cv2.putText(frame, label, (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 150, (0, 255, 0), 2)
            #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 1)
            frame_id += 1
            self.result.append(frame)
            print('len result', len(self.result))


if __name__ == '__main__':

    path_in = 'input.avi'  # Đường dẫn đến file video
    path_out = 'output.avi'  # Đường dẫn lưu file video
    if len(sys.argv) >= 2:
        path_in = sys.argv[1]
    if len(sys.argv) >= 3:
        path_out = sys.argv[2]
    object_tracking = ObjectTracking()
    cap = cv2.VideoCapture(path_in)
    out = None

    th = Thread(target=object_tracking.tracking, args=[cap])
    th.start()
    time.sleep(5)
    while len(object_tracking.result) > 0 or th.isAlive:
        if len(object_tracking.result) > 0:
            frame = object_tracking.result.pop(0)
            cv2.imshow('frame', frame)
            if out is None:
                out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                      (frame.shape[1], frame.shape[0]))
            out.write(frame)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    object_tracking.stop()
    out.release()
