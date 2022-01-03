import cv2
import os
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate


class CoralCam(object):
    __instance = None

    def __new__(cls):
        if CoralCam.__instance is None:
            CoralCam.__instance = object.__new__(cls)
        CoralCam.__instance.video = cv2.VideoCapture(0)
        CoralCam.__instance.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        CoralCam.__instance.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        CoralCam.__instance.inference_type = None
        CoralCam.__instance.engine = None
        return CoralCam.__instance

    # def __init__(self):
    #     self.video = cv2.VideoCapture(0)
    #     self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #     self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #     self.inference_type = None
    #     self.engine = None

    def __del__(self):
        self.video.release()

    def set_engine(self, inference_type, model):
        print(f'Switching\n - inference type: {inference_type}\n - model: {model}')
        self.__instance.inference_type = inference_type
        self.__instance.engine = Interpreter(
            os.path.join('test_data', 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'),
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        # self.__instance.engine = Interpreter(model, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        if success:
            ret, jpeg = cv2.imencode('.jpg', image)
            if ret:
                return jpeg.tobytes()
        return None
