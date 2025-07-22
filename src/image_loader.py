from typing import List
import numpy as np
import cv2


def load_img(img_path:str) -> np.array:
    return cv2.imread(img_path)

def load_labels(labels_path:str) -> List:
    labels = []
    with open(labels_path, 'rb') as file:
        for line in file.readlines():
            labels.append(list(map(float, line.decode('utf-8').strip().split(' '))))
    return labels

def show_img(img: np.array) -> None:
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_img_with_bbox(img:np.array, labels:List) -> np.array:
    height, width = img.shape[:2]
    for label in labels:
        c, x, y, w, h = label
        p1 = int((x-w/2)*width), int((y-h/2)*height)
        p2 = int((x+w/2)*width), int((y+h/2)*height)
        cv2.rectangle(img, p1, p2, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def show_img_with_bbox(img_path:str, labels_path:str) -> None:
    img = load_img(img_path)
    labels = load_labels(labels_path)
    img = get_img_with_bbox(img, labels)
    show_img(img)