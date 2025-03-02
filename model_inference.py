import tensorflow as tf
import cv2  
from config import Config

class ModelInference:
    def __init__(self):
        loaded_model = tf.saved_model.load(Config.MODEL_PATH)
        self.model = loaded_model.signatures['serving_default']

    def preprocess_frame(self, frame):
        frame_resized = cv2.resize(frame, Config.INPUT_SIZE)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.int32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        return input_tensor

    def run_inference(self, frame):
        input_image = self.preprocess_frame(frame)
        outputs = self.model(input_image)
        keypoints = outputs['output_0'].numpy()[0, 0]
        return keypoints