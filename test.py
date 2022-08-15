import xml.etree.ElementTree as ET
import time
from pathlib import Path

import numpy as np
from absl import app, flags, logging

from core import utils
from core.utils import bbox_giou
from detect import detector

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within images')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):
        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


def convert_voc_to_coco(boxes):
    bboxs = np.zeros((50, 4), np.float32)
    classes = np.zeros(50, np.float32)
    confidence = np.zeros(50, np.float32)
    for i, box in enumerate(boxes):
        bboxs[i] = box
        confidence[i] = 1.
    return [bboxs, confidence, classes, len(boxes)]


def main(_argv):
    dataset_path = Path("data/dataset/test")
    annotations_path = dataset_path / 'annotations'
    images_path = dataset_path / 'images'
    images = []
    annotations = []
    for i, annotation_xml in enumerate(list(annotations_path.glob('**/*'))):
        image_name, boxes = read_content(str(annotation_xml))
        image_path = images_path / image_name
        images.append(str(image_path))
        annotations.append(convert_voc_to_coco(boxes))
        if i == 3:
            break
    start = time.time()

    predictions = detector(images, annotations)
    total_time = time.time() - start

    tp, fn, fp = 0, 0, 0
    for prediction, annotation in zip(predictions, annotations):
        list_visualized_index = []
        aux_max_iou_list = np.zeros(50)
        counter = 0
        for label in annotation[0]:
            if sum(label) != 0:
                counter += 1
                label_grid = np.zeros((50, 4))
                label_grid[:, :] = label
                giou = bbox_giou(prediction[0], annotation[0])
                numpy_list_giou = list(giou.numpy())
                new_index = numpy_list_giou.index(max(numpy_list_giou))
                if new_index not in list_visualized_index and max(numpy_list_giou) > .5:
                    list_visualized_index.append(new_index)
                    aux_max_iou_list[new_index] = max(numpy_list_giou)

                elif max(numpy_list_giou) > aux_max_iou_list[new_index] and max(numpy_list_giou) > .5:
                    aux_max_iou_list[new_index] = max(numpy_list_giou)
                    list_visualized_index.pop(-1)
                    list_visualized_index.append(new_index)

        counter_aux = 0
        for prediction_count in prediction[0]:
            if sum(prediction_count) != 0:
                counter_aux += 1

        tp += len(list_visualized_index)
        fn += (counter - len(list_visualized_index))
        fp += (counter_aux - len(list_visualized_index))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    time_per_image = total_time / len(predictions)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Avg time per image: " + str(time_per_image))
    print("TP: " + str(tp))
    print("FN: " + str(fn))
    print("FP: " + str(fp))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
