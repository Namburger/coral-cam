import cv2
import os

import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

from model_utils import ModelUtils


class InferenceAdaptor:
    coral_bgr = (77, 94, 253)

    @staticmethod
    def classify(interpreter, image):
        input_details = interpreter.get_input_details()
        width = input_details[0]['shape'][2]
        height = input_details[0]['shape'][1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height))
        input_data = np.expand_dims(resized, axis=0)

        # Set input and run inference.
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output. 
        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(interpreter.get_tensor(output_details['index']))
        # If the model is quantized (uint8 data), then dequantize the results
        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)
        max_idx = np.argmax(output)
        score = output[max_idx]
        class_label = f'class: {ModelUtils.get_classification_class(max_idx)}'
        label_size, _ = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_x_location = 1280 - ((label_size[0]) + 30)
        label_y_location = label_size[1] + 5
        cv2.putText(image, class_label, (label_x_location, label_y_location), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    InferenceAdaptor.coral_bgr, 2)
        score_label = f'score: {score}'
        score_size, _ = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        score_x_location = label_x_location
        score_y_location = label_y_location + score_size[1] + 5
        cv2.putText(image, score_label, (score_x_location, score_y_location), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    InferenceAdaptor.coral_bgr, 2)
        return image

    @staticmethod
    def detect(interpreter, image, image_width, image_height):
        input_details = interpreter.get_input_details()
        width = input_details[0]['shape'][2]
        height = input_details[0]['shape'][1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height))
        input_data = np.expand_dims(resized, axis=0)

        # Set input and run inference.
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output tensor
        output_details = interpreter.get_output_details()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i in range(len(scores)):
            if 0.5 < scores[i] < 1.0:
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be
                # within image using max() and min()
                y_min = int(max(1, (boxes[i][0] * image_height)))
                x_min = int(max(1, (boxes[i][1] * image_width)))
                y_max = int(min(image_height, (boxes[i][2] * image_height)))
                x_max = int(min(image_width, (boxes[i][3] * image_width)))
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), InferenceAdaptor.coral_bgr, 4)
                label = '%s: %d%%' % (ModelUtils.get_detection_class(int(classes[i])), int(scores[i] * 100))
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Make sure not to draw label too close to top of window.
                label_y = max(y_min, label_size[1] + 10)
                cv2.rectangle(image, (x_min, label_y - label_size[1] - 10), (
                    x_min + label_size[0], label_y + base_line - 10), InferenceAdaptor.coral_bgr, cv2.FILLED)
                cv2.putText(image, label, (x_min, label_y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image

    @staticmethod
    def pose_estimate(interpreter, image, image_width, image_height):
        return image


class CoralCam(object):
    __instance = None

    # CoralCam is a singleton class that give frame by frame image to the html.
    def __new__(cls):
        if CoralCam.__instance is None:
            CoralCam.__instance = object.__new__(cls)
        CoralCam.__instance.video = cv2.VideoCapture(0)
        CoralCam.__instance.width = 1280
        CoralCam.__instance.video.set(cv2.CAP_PROP_FRAME_WIDTH, CoralCam.__instance.width)
        CoralCam.__instance.height = 720
        CoralCam.__instance.video.set(cv2.CAP_PROP_FRAME_HEIGHT, CoralCam.__instance.height)
        CoralCam.__instance.current_model = None  # Stores current model path.
        CoralCam.__instance.current_model_size = None  # Stores current model size as a string.
        CoralCam.__instance.inference_type = None  # [classification, detection, pose-estimation]
        CoralCam.__instance.engine = None  # Inference Engine
        return CoralCam.__instance

    def __del__(self):
        self.video.release()

    def set_engine(self, inference_type, model):
        self.__instance.current_model = ModelUtils.get_model_path(model)
        print(f'Mode: {inference_type}'
              f'\n - model name: {model}'
              f'\n - model path: {self.__instance.current_model}')
        self.__instance.inference_type = inference_type
        self.__instance.engine = Interpreter(
            self.__instance.current_model,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        self.__instance.engine.allocate_tensors()
        input_details = self.__instance.engine.get_input_details()
        width = input_details[0]['shape'][2]
        height = input_details[0]['shape'][1]
        self.__instance.current_model_size = f'{width}x{height}'

    def add_model_info(self, image):
        model_name = str(self.__instance.current_model).split('/')[-1]
        model_name_size, _ = cv2.getTextSize(model_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        model_name_y = model_name_size[1] + 5
        cv2.putText(image, model_name, (10, model_name_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    InferenceAdaptor.coral_bgr, 1)
        model_size, _ = cv2.getTextSize(self.__instance.current_model_size, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        model_size_y = model_name_y + model_size[1] + 5
        cv2.putText(image, self.__instance.current_model_size, (10, model_size_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    InferenceAdaptor.coral_bgr, 1)

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        if success:
            if self.__instance.inference_type == 'classification':
                image = InferenceAdaptor.classify(CoralCam.__instance.engine, image)
            elif self.__instance.inference_type == 'detection':
                image = InferenceAdaptor.detect(CoralCam.__instance.engine, image, CoralCam.__instance.width,
                                                CoralCam.__instance.height)
            else:  # pose-estimation
                image = InferenceAdaptor.pose_estimate(CoralCam.__instance.engine, image, CoralCam.__instance.width,
                                                       CoralCam.__instance.height)
            self.add_model_info(image)
            ret, jpeg = cv2.imencode('.jpg', image)
            if ret:
                return jpeg.tobytes()
        return None
