from scipy.io import loadmat
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys


def show(img, name = "window"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1600, 1600)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw(img, xmin, ymin, width, height):
    return cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmin)+int(width), int(ymin)+int(height)), (0, 255, 0), 2)


def save_yolo_format(filename, bboxes, img_width, img_height):
    with open(Path(filename).with_suffix(".txt"), 'w') as f:
        for bbox in bboxes:
            arr = np.array(bbox )
            if np.any(arr < 0) or (np.count_nonzero(arr) == 0) :
                # print("NEGATIVE VALUES")
                continue
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            assert x_center >= 0 and y_center >= 0 and width > 0 and height > 0
            f.write(f"0 {x_center} {y_center} {width} {height}\n")

def save_yolo_image(filename, img):
    cv2.imwrite(Path(filename).with_suffix(".jpg").__str__(), img)



def process_frame(gt_mat_path, showing=False, strict=True):
    if gt_mat_path in [Path("ObjectGT/MVI_1584_VIS_ObjectGT.mat"), Path("ObjectGT/MVI_0799_VIS_OB_ObjectGT.mat"), Path("ObjectGT/MVI_0790_VIS_OB_ObjectGT.mat")]: return
    identifier = gt_mat_path.stem.replace("_ObjectGT", "")
    video_path = "Videos" / Path(gt_mat_path.name.replace("_ObjectGT", "")).with_suffix(".avi")
    vidcap = cv2.VideoCapture(video_path.__str__())
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    data = loadmat(gt_mat_path)
    bb_data = data["structXML"]["BB"]
    bb_data = np.array(bb_data).flatten()
    if strict:
        assert video_length == len(bb_data), f"Problem file: {video_path}, video={video_length}, bbdata={len(bb_data)}"

    for i, elem in enumerate(bb_data):
        success, image = vidcap.read()
        if success and elem.any():
            bboxes = list()
            for index in range(len(elem)):
                xmin, ymin, width, height = elem[index][0], elem[index][1], elem[index][2], elem[index][3]
                bboxes.append((xmin, ymin, width, height))

            if showing:
                print(bboxes)
                sys.stdout.flush()
                [draw(image, elem[0], elem[1], elem[2], elem[3]) for elem in bboxes]
                show(image)
            else:
                save_yolo_format((label_path /( identifier.__str__() + f"_{i:06}")).__str__(), bboxes, image.shape[1], image.shape[0])
                save_yolo_image( (saved_image_path /( identifier.__str__() + f"_{i:06}")).__str__(), image)


if __name__ == "__main__":
    ground_truth_path = Path("ObjectGT")
    label_path = Path("labels")
    saved_image_path = Path("images")
    label_path.mkdir(exist_ok=True)
    saved_image_path.mkdir(exist_ok=True)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_frame, ground_truth_path.iterdir()), total=len(list(ground_truth_path.iterdir()))))


