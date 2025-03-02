import time
import cv2
from video_capture import VideoCapture
from model_inference import ModelInference
from fall_detection import FallDetection
from config import Config

def main():
    video = VideoCapture(Config.VIDEO_SOURCE)  
    model = ModelInference()
    fall_detector = FallDetection()

    fall_detected = False
    fps = video.get_fps()
    delay = int(1000 / fps) if fps > 0 else 1

    try:
        while True:
            frame = video.get_frame()
            if frame is None:
                print("End of video or cannot read frame.")
                break

            keypoints = model.run_inference(frame)
            is_fall, bbox = fall_detector.detect_fall(keypoints, frame.shape)

            if bbox:
                print("Person detected")
                min_x, min_y, max_x, max_y = bbox
                if is_fall:
                    print("Fall detected!")
                    color = (0, 0, 255) 
                    fall_detected = True
                else:
                    print("No fall detected.")
                    color = (0, 255, 0)  
                    fall_detected = False
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
            else:
                print("No person detected.")

            cv2.imshow('Fall Detection', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()