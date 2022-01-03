import cv2
import os

import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate


class InferenceAdaptor:
    @staticmethod
    def detection(interpreter, image, image_width, image_height):
        input_details = interpreter.get_input_details()
        width = input_details[0]['shape'][2]
        height = input_details[0]['shape'][1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height))
        input_data = np.expand_dims(resized, axis=0)

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
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (10, 255, 0), 4)
                label = '%s: %d%%' % (classes[i], int(scores[i] * 100))
                label_size, base_line = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Make sure not to draw label too close to top of window
                label_location = max(y_min, label_size[1] + 10)
                cv2.rectangle(image, (x_min, label_location - label_size[1] - 10), (
                    x_min + label_size[0], label_location + base_line - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (x_min, label_location - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
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
        CoralCam.__instance.inference_type = None  # [classification, detection, pose-estimation]
        CoralCam.__instance.engine = None  # Inference Engine
        return CoralCam.__instance

    def __del__(self):
        self.video.release()

    def set_engine(self, inference_type, model):
        print(f'Switching\n - inference type: {inference_type}\n - model: {model}')
        self.__instance.inference_type = inference_type
        self.__instance.engine = Interpreter(
            os.path.join('test_data', 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'),
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        self.__instance.engine.allocate_tensors()
        # self.__instance.engine = Interpreter(model, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        if success:
            image = InferenceAdaptor.detection(CoralCam.__instance.engine, image, CoralCam.__instance.width,
                                               CoralCam.__instance.height)
            ret, jpeg = cv2.imencode('.jpg', image)
            if ret:
                return jpeg.tobytes()
        return None
