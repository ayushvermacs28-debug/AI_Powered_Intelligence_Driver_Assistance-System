import os
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

# ==========================================
# DATASET PATH
# ==========================================

BASE_FOLDER = r"C:\Users\mridu\Downloads\yolo_pothole\pothole dataset\pothole ka dataseeettt"

IMAGE_FOLDER = os.path.join(BASE_FOLDER, "images")
ANNOTATION_FOLDER = os.path.join(BASE_FOLDER, "annotations")

OUTPUT_FOLDER = r"C:\Users\mridu\Downloads\yolo_pothole\pothole_yolo"


# ==========================================
# CREATE FOLDERS
# ==========================================

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(OUTPUT_FOLDER, "images", split),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(OUTPUT_FOLDER, "labels", split),
        exist_ok=True
    )


# ==========================================
# GET IMAGES
# ==========================================

images = []

for file in os.listdir(IMAGE_FOLDER):

    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        images.append(file)

print("Total images:", len(images))


# ==========================================
# SHUFFLE
# ==========================================

random.seed(42)
random.shuffle(images)


# ==========================================
# SPLIT DATA
# 70% TRAIN
# 20% VALIDATION
# 10% TEST
# ==========================================

total = len(images)

train_count = int(total * 0.70)
val_count = int(total * 0.20)

train_images = images[:train_count]

val_images = images[
    train_count:train_count + val_count
]

test_images = images[
    train_count + val_count:
]

print("Train:", len(train_images))
print("Validation:", len(val_images))
print("Test:", len(test_images))


# ==========================================
# XML TO YOLO
# ==========================================

def convert_xml_to_yolo(xml_file, txt_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text.strip()

        # pothole = class 0
        if class_name != "pothole":
            continue

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Keep coordinates inside image
        xmin = max(0, min(xmin, width))
        xmax = max(0, min(xmax, width))

        ymin = max(0, min(ymin, height))
        ymax = max(0, min(ymax, height))

        # Ignore invalid boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        box_width = xmax - xmin
        box_height = ymax - ymin

        # Normalize
        x_center = x_center / width
        y_center = y_center / height

        box_width = box_width / width
        box_height = box_height / height

        line = (
            f"0 "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        yolo_lines.append(line)

    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_lines))


# ==========================================
# PROCESS TRAIN / VAL / TEST
# ==========================================

def process_split(image_list, split):

    print("\nProcessing:", split)

    for image_name in image_list:

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_name
        )

        base_name = os.path.splitext(image_name)[0]

        xml_path = os.path.join(
            ANNOTATION_FOLDER,
            base_name + ".xml"
        )

        # Check XML
        if not os.path.exists(xml_path):
            print("Missing XML:", image_name)
            continue

        # Check image
        try:
            img = Image.open(image_path)
            img.verify()

        except Exception:
            print("Bad image:", image_name)
            continue

        # Destination image
        destination_image = os.path.join(
            OUTPUT_FOLDER,
            "images",
            split,
            image_name
        )

        # Destination label
        destination_label = os.path.join(
            OUTPUT_FOLDER,
            "labels",
            split,
            base_name + ".txt"
        )

        # Copy image
        shutil.copy2(
            image_path,
            destination_image
        )

        # Convert XML → YOLO TXT
        convert_xml_to_yolo(
            xml_path,
            destination_label
        )

    print(split, "completed.")


# ==========================================
# RUN
# ==========================================

process_split(train_images, "train")

process_split(val_images, "val")

process_split(test_images, "test")


# ==========================================
# CREATE data.yaml
# ==========================================

yaml_content = f"""path: {OUTPUT_FOLDER}

train: images/train
val: images/val
test: images/test

names:
  0: pothole
"""

yaml_path = os.path.join(
    OUTPUT_FOLDER,
    "data.yaml"
)

with open(yaml_path, "w") as f:
    f.write(yaml_content)


print("\n================================")
print("PREPROCESSING COMPLETED")
print("================================")
print("Output:", OUTPUT_FOLDER)
print("YAML:", yaml_path)