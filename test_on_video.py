import cv2
from yolo.yolo_opencv import ObjectDetection
from deepsort import *
import sys

if __name__ == '__main__':

    path_in = 'input.avi'  # Đường dẫn đến file video
    path_out = 'output.avi'  # Đường dẫn lưu file video
    if len(sys.argv) >= 2:
        path_in = sys.argv[1]
    if len(sys.argv) >= 3:
        path_out = sys.argv[2]
    object_detection = ObjectDetection()
    cap = cv2.VideoCapture(path_in)

    # Initialize deep sort.
    deepsort = deepsort_rbc()

    frame_id = 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path_out, fourcc, 10.0, (1920, 1080))

    while True:
        print(frame_id)

        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        if ret is False:
            frame_id += 1
            break

        objects = object_detection.detect(frame)
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

        tracker, detections_class = deepsort.run_deep_sort(frame, out_scores, detections, labels)

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

        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1
