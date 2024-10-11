import cv2
import json
import threading
import numpy as np
import colorama as col
from djitellopy import Tello
from FlyLib3.vision.apriltag import ApriltagDetector, ApriltagDetectionResult
from FlyLib3.math.pid import PID

drone = Tello()
capture = cv2.VideoCapture(0)
apriltag_detector = ApriltagDetector()
last_detected_apriltag = None
current_detected_apriltag = None
align_pid = PID(0.1, 0, 0)
alignment_error_px = 40

def on_apriltag_detected(apriltag: ApriltagDetectionResult, frame: np.ndarray):
    # TODO: Checar direcciones
    match apriltag.tag_id:
        case 1:
            print("Apriltag 1 detected")
            drone.move_forward(80)
        case 2:
            print("Apriltag 2 detected")
            drone.move_left(80)
        case 3:
            print("Apriltag 3 detected")
            drone.move_right(80)
        case 4:
            print("Apriltag 4 detected")
            drone.land()

def loop():
    global last_detected_apriltag
    global current_detected_apriltag

    while True:
        ret, frame = capture.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = apriltag_detector.detect(gray)

        if len(detections) > 0:
            # Get the first apriltag detected
            detection = detections[0]

            # Draw the tag bounding box
            cv2.polylines(frame, [np.int32(detection.corners)], True, (0, 255, 0), 2)

            # Draw the tag id
            cv2.putText(frame, "DETECTED ID " + str(detection.tag_id), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Handle events
            if (last_detected_apriltag == None and current_detected_apriltag != None) or last_detected_apriltag.tag_id != current_detected_apriltag:
                on_apriltag_detected(detection, frame)
        
            # Draw other apriltags in the frame
            for detection in detections[1:]:
                cv2.polylines(frame, [np.int32(detection.corners)], True, (255, 0, 0), 2)

            last_detected_apriltag = current_detected_apriltag
            current_detected_apriltag = detection

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    # Configure drone
    print(col.Style.BRIGHT + "Connecting to drone..." + col.Style.RESET_ALL)
    while not drone.get_current_state():
        try:
            drone.connect()
            print(col.Fore.GREEN + "Drone connected!" + col.Style.RESET_ALL)
            break
        except Exception as e:
            print(col.Fore.YELLOW + "Could not connect to drone, retrying..." + col.Style.RESET_ALL)
            print(e)
            continue
    
    drone.streamon()
    drone.set_speed(25)
    drone.takeoff()

    # Start loop thread
    threading.Thread(target=loop, daemon=True).start()

if __name__ == '__main__':
    main()