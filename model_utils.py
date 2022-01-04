import os


def read_detection_label():
    coco_label_path = os.path.join('test_data', 'coco_labels.txt')
    with open(coco_label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            return {}
        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def read_classification_label():
    imagenet_label_path = os.path.join('test_data', 'imagenet_labels.txt')
    with open(imagenet_label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


class ModelUtils:
    model_name_to_path = {
        'MobileNet V1': os.path.join('test_data', 'mobilenet_v1_1.0_224_quant_edgetpu.tflite'),
        'MobileNet V2': os.path.join('test_data', 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'),
        'Inception V1': os.path.join('test_data', 'inception_v1_224_quant_edgetpu.tflite'),
        'Inception V2': os.path.join('test_data', 'inception_v2_224_quant_edgetpu.tflite'),
        'Inception V3': os.path.join('test_data', 'inception_v3_299_quant_edgetpu.tflite'),
        'Inception V4': os.path.join('test_data', 'inception_v4_299_quant_edgetpu.tflite'),
        'ResNet-50': os.path.join('test_data', 'tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite'),
        'EfficientNet (S)': os.path.join('test_data', 'efficientnet-edgetpu-S_quant_edgetpu.tflite'),
        'EfficientNet (M)': os.path.join('test_data', 'efficientnet-edgetpu-M_quant_edgetpu.tflite'),
        'EfficientNet (L)': os.path.join('test_data', 'efficientnet-edgetpu-L_quant_edgetpu.tflite'),
        'SSD MobileNet V1': os.path.join('test_data', 'ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite'),
        'SSD MobileNet V2': os.path.join('test_data', 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'),
        'SSDLite MobileDet': os.path.join('test_data', 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'),
        'EfficientDet-Lite0': os.path.join('test_data', 'efficientdet_lite0_320_ptq_edgetpu.tflite'),
        'EfficientDet-Lite1': os.path.join('test_data', 'efficientdet_lite1_384_ptq_edgetpu.tflite'),
        'EfficientDet-Lite2': os.path.join('test_data', 'efficientdet_lite2_448_ptq_edgetpu.tflite'),
        'EfficientDet-Lite3': os.path.join('test_data', 'efficientdet_lite3_512_ptq_edgetpu.tflite'),
        'PoseNet MobileNet V1 (S)': os.path.join('test_data', 'posenet',
                                                 'posenet_mobilenet_v1_075_353_481_16_quant_decoder_edgetpu.tflite'),
        'PoseNet MobileNet V1 (M)': os.path.join('test_data', 'posenet',
                                                 'posenet_mobilenet_v1_075_481_641_16_quant_decoder_edgetpu.tflite'),
        'PoseNet MobileNet V1 (L)': os.path.join('test_data', 'posenet',
                                                 'posenet_mobilenet_v1_075_721_1281_16_quant_decoder_edgetpu.tflite'),
        'MoveNet.SinglePose.Lightning': os.path.join('test_data', 'movenet_single_pose_lightning_ptq_edgetpu.tflite'),
        'MoveNet.SinglePose.Thunder': os.path.join('test_data', 'movenet_single_pose_thunder_ptq_edgetpu.tflite')
    }

    detection_label = read_detection_label()
    classification_label = read_classification_label()

    @staticmethod
    def get_model_path(model_name, edgetpu=True):
        if edgetpu:
            return ModelUtils.model_name_to_path[model_name]
        else:
            return ModelUtils.model_name_to_path[model_name].replace('_edgetpu', '')

    @staticmethod
    def get_detection_class(key):
        return ModelUtils.detection_label[key]

    @staticmethod
    def get_classification_class(key):
        return ModelUtils.classification_label[key]
