import cv2

class VideoCapture:
    def __init__(self, video_source="01.mp4"):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open video source {video_source}")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)