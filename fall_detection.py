import numpy as np
from config import Config

class FallDetection:
    @staticmethod
    def calculate_angle(p1, p2, p3):
        vec1 = p1 - p2
        vec2 = p3 - p2
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 180.0
        cos_theta = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        return angle

    def detect_fall(self, keypoints, frame_shape):
        valid_keypoints = [kp for kp in keypoints if kp[2] > Config.CONFIDENCE_THRESHOLD]
        if len(valid_keypoints) < 5:  # Relaxed requirement
            return False, None

        # Scale keypoints to frame size
        xs = [kp[1] * frame_shape[1] for kp in valid_keypoints]
        ys = [kp[0] * frame_shape[0] for kp in valid_keypoints]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        if width == 0:
            return False, None
        ratio = height / width

        # Get key body parts
        left_hip = keypoints[11]
        left_knee = keypoints[13]
        left_ankle = keypoints[15]
        right_hip = keypoints[12]
        right_knee = keypoints[14]
        right_ankle = keypoints[16]

        # Calculate knee angles
        left_knee_angle = None
        if all(kp[2] > Config.CONFIDENCE_THRESHOLD for kp in [left_hip, left_knee, left_ankle]):
            left_knee_angle = self.calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])

        right_knee_angle = None
        if all(kp[2] > Config.CONFIDENCE_THRESHOLD for kp in [right_hip, right_knee, right_ankle]):
            right_knee_angle = self.calculate_angle(right_hip[:2], right_knee[:2], right_ankle[:2])

        # Fall detection logic
        if ratio < Config.HEIGHT_WIDTH_RATIO_THRESHOLD:
            if (left_knee_angle is None or left_knee_angle > Config.KNEE_ANGLE_THRESHOLD) and \
               (right_knee_angle is None or right_knee_angle > Config.KNEE_ANGLE_THRESHOLD):
                return True, (int(min_x), int(min_y), int(max_x), int(max_y))
        return False, (int(min_x), int(min_y), int(max_x), int(max_y))