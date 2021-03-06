import os
import numpy as np
import cv2
import eel
from time import time

import tflite_runtime.interpreter
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from model_utils import ModelUtils

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
POSENET_SHARED_LIB = os.path.join(
    'posenet_lib', os.uname().machine, 'posenet_decoder.so')


class InferenceAdaptor:
    """
    This class acts as a namespace that uses the adapter pattern to extends coral cam to do detection,
    classification, pose estimation, and segmentation on an image.
    """
    coral_bgr = (77, 94, 253)  # The coral color in (B, G, R).

    @staticmethod
    def add_model_info(interpreter: tflite_runtime.interpreter.Interpreter, model_path: str, latency: str,
                       image: cv2.cvtColor):
        """ Writes the model info on the top left corner of an image.
        :param interpreter: The tflite interpreter.
        :param model_path: The path to the model.
        :param latency: The latency string to put on the image.
        :param image: The image to put the latency string on.
        :return: None
        """
        latency_size, _ = cv2.getTextSize(latency, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(image, latency, (10, latency_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    InferenceAdaptor.coral_bgr, 1)
        model_name = f'model: {model_path.split("/")[-1]}'
        model_name_size, _ = cv2.getTextSize(model_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        model_name_y = model_name_size[1] + 20
        cv2.putText(image, model_name, (10, model_name_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    InferenceAdaptor.coral_bgr, 1)

        input_details = interpreter.get_input_details()
        width = input_details[0]['shape'][2]
        height = input_details[0]['shape'][1]
        size_str = f'size: {width}x{height}'
        model_size, _ = cv2.getTextSize(size_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        model_size_y = model_name_y + model_size[1] + 5
        cv2.putText(image, size_str, (10, model_size_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    InferenceAdaptor.coral_bgr, 1)

    @staticmethod
    def run_inference(interpreter: tflite_runtime.interpreter.Interpreter, image: cv2.cvtColor):
        """ Transform image input into input tensors for the interpreter and then call invoke.
        :param interpreter: The tflite interpreter.
        :param image: The image.
        :return: None
        """
        input_details = interpreter.get_input_details()
        width = input_details[0]['shape'][2]
        height = input_details[0]['shape'][1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height))
        if interpreter.get_input_details()[0]['dtype'] == np.float32:
            input_data = np.float32(resized) / 128.0 - 1.0
        else:
            input_data = np.asarray(resized)
        input_data = np.expand_dims(input_data, axis=0)

        # Set input and run inference.
        t0 = time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        latency = 'latency: {:.2f} ms'.format((time() - t0) * 1000)
        return width, height, latency

    @staticmethod
    def classify(interpreter: tflite_runtime.interpreter.Interpreter, image: cv2.cvtColor, model_name: str):
        """ Run the classification on the image and then writes labels on the image.
        :param interpreter: The tflite interpreter.
        :param image: The image to run classification on.
        :param model_name: The path to the model.
        :return: The processed image.
        """
        # Run Inference on image.
        _, _, latency = InferenceAdaptor.run_inference(interpreter, image)

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
        InferenceAdaptor.add_model_info(interpreter, model_name, latency, image)
        return image

    @staticmethod
    def detect(interpreter: tflite_runtime.interpreter.Interpreter, image: cv2.cvtColor, model_name: str):
        """ Run detection on the image and then writes labels on the image.
        :param interpreter: The tflite interpreter.
        :param image: The image to run detection on.
        :param model_name: The path to the model.
        :return: The processed image.
        """
        # Run Inference on image.
        _, _, latency = InferenceAdaptor.run_inference(interpreter, image)

        # Get output tensor
        image_height, image_width = image.shape[0], image.shape[1]
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
        InferenceAdaptor.add_model_info(interpreter, model_name, latency, image)
        return image

    @staticmethod
    def pose_estimate(interpreter: tflite_runtime.interpreter.Interpreter, image: cv2.cvtColor, model_name: str):
        """ Run detection on the image and then writes labels on the image.
       :param interpreter: The tflite interpreter.
       :param image: The image to run pose estimation on.
       :param model_name: The name of the model.
       :return: The processed image.
       """
        # Run Inference on image.
        model_width, model_height, latency = InferenceAdaptor.run_inference(interpreter, image)

        def get_output_tensor(interpreter_, idx):
            return np.squeeze(interpreter_.tensor(interpreter.get_output_details()[idx]['index'])())

        image_height, image_width = image.shape[0], image.shape[1]
        if 'posenet' in model_name:
            keypoints = get_output_tensor(interpreter, 0)
            # keypoints_scores = get_output_tensor(interpreter, 1)
            # pose_scores = get_output_tensor(interpreter, 2)
            num_poses = get_output_tensor(interpreter, 3)

            for i in range(int(num_poses)):
                for keypoint in keypoints[i]:
                    y, x = keypoint
                    x, y = int((x / model_width) * image_width), int((y / model_height) * image_height)
                    image = cv2.circle(image, (x, y), radius=3, color=InferenceAdaptor.coral_bgr, thickness=5)
        else:
            keypoints = get_output_tensor(interpreter, 0)
            for keypoint in keypoints:
                y, x, score = keypoint
                if 0.5 < score < 1.0:
                    x, y = int(x * image_width), int(y * image_height)
                    image = cv2.circle(image, (x, y), radius=3, color=InferenceAdaptor.coral_bgr, thickness=5)
        InferenceAdaptor.add_model_info(interpreter, model_name, latency, image)
        return image

    @staticmethod
    def segmentation(interpreter: tflite_runtime.interpreter.Interpreter, image: cv2.cvtColor, model_name: str):
        """ Run detection on the image and then writes labels on the image.
        :param interpreter: The tflite interpreter.
        :param image: The image to run segmentation on.
        :param model_name: The path to the model.
        :return: The processed image.
        """
        # Run Inference on image.
        _, _, latency = InferenceAdaptor.run_inference(interpreter, image)

        # Get output.
        image_height, image_width = image.shape[0], image.shape[1]
        output_details = interpreter.get_output_details()[0]
        output = interpreter.tensor(output_details['index'])()[0].astype(np.uint8)
        if len(output.shape) == 3:
            output = np.argmax(output, axis=-1)
        mask_img = Image.fromarray(ModelUtils.label_to_color_image(output).astype(np.uint8))
        image = cv2.resize(cv2.cvtColor(np.array(mask_img), cv2.COLOR_RGB2BGR), dsize=(image_width, image_height))
        InferenceAdaptor.add_model_info(interpreter, model_name, latency, image)
        return image


class CoralCam(object):
    """ CoralCam is a singleton class that essentially acts as a normal camera with added AI functionality."""
    __instance = None  # The Coral Cam instance.

    def __new__(cls):
        """ Constructor for the CoralCam singleton. """
        if CoralCam.__instance is None:
            CoralCam.__instance = object.__new__(cls)
        CoralCam.__instance.video = cv2.VideoCapture(0)  # The cv2 camera capture.
        CoralCam.__instance.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        CoralCam.__instance.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        CoralCam.__instance.current_model = None  # Stores current model path.
        CoralCam.__instance.inference_type = None  # [classification, detection, pose-estimation]
        CoralCam.__instance.engine = None  # Inference Engine
        return CoralCam.__instance

    def __del__(self):
        """ Destructor. """
        self.__instance.video.release()

    def set_engine(self, inference_type, model: str, edgetpu: bool):
        """ Switch the inference engine.
        :param inference_type: The inference mode ['classification', 'detection', 'pose-estimation', 'segmentation'].
        :param model: The name of the model.
        :param edgetpu: Whether to use the edgetpu or not.
        :return: None
        """
        current_model = ModelUtils.get_model_path(model, edgetpu)
        if edgetpu:
            try:
                if 'posenet' in current_model:
                    self.__instance.engine = Interpreter(
                        current_model,
                        experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB), load_delegate(POSENET_SHARED_LIB)])
                else:
                    self.__instance.engine = Interpreter(
                        current_model,
                        experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)])
            except Exception as e:
                msg = f'Failed to switch to edgetpu model, reason: {e}'
                eel.updateLog(msg)()
                return
        else:
            if 'posenet' in current_model:
                self.__instance.engine = Interpreter(current_model,
                                                     experimental_delegates=[load_delegate(POSENET_SHARED_LIB)])
            else:
                self.__instance.engine = Interpreter(current_model)

        self.__instance.current_model = ModelUtils.get_model_path(model, edgetpu)
        self.__instance.inference_type = inference_type

        # Initialize new model.
        self.__instance.engine.allocate_tensors()

        msg = f'Mode: {self.__instance.inference_type} - model name: {model} - model path: {self.__instance.current_model} '
        eel.updateLog(msg)()

    def get_frame(self):
        """ Captures the image from the camera, run inference and label the image.
        :return: The processed image.
        """
        success, image = self.__instance.video.read()
        if success:
            if self.__instance.inference_type == 'classification':
                image = InferenceAdaptor.classify(self.__instance.engine, image, self.__instance.current_model)
            elif self.__instance.inference_type == 'detection':
                image = InferenceAdaptor.detect(self.__instance.engine, image, self.__instance.current_model)
            elif self.__instance.inference_type == 'pose-estimation':
                image = InferenceAdaptor.pose_estimate(self.__instance.engine, image, self.__instance.current_model)
            else:
                image = InferenceAdaptor.segmentation(self.__instance.engine, image, self.__instance.current_model)

            # We are using Motion JPEG, but OpenCV defaults to capture raw images,
            # so we must encode it into JPEG in order to correctly display the
            # video stream.
            ret, jpeg = cv2.imencode('.jpg', image)
            if ret:
                return jpeg.tobytes()
        return None
