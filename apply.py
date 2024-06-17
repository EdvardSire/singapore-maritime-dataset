from pathlib import Path
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def draw_bboxes(image_path, label_path, output_path):
    image = cv2.imread(str(image_path))

    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        _, x_center, y_center, width, height = map(float, label.strip().split())

        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imwrite(str(output_path), image)


def pframe(image_path):
    label_path = label_dir / image_path.with_suffix(".txt").name
    draw_bboxes(image_path, label_path, (output_dir / image_path.name))




if __name__ == "__main__":
    image_dir = Path('images')
    label_dir = Path('labels')

    output_dir = Path('output_images')
    output_dir.mkdir(exist_ok=True)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(pframe, list(image_dir.iterdir()))))
