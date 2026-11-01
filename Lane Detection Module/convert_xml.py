import os
import shutil
import random
import xml.etree.ElementTree as ET

IMAGE_DIR = r"C:\Users\Ananya\Downloads\LANE DETECTION\LANE DETECTION\road_line_images\road_line_images"
XML_DIR = r"C:\Users\Ananya\Downloads\LANE DETECTION\LANE DETECTION\road_line_annotation\road_line_annotation"

BASE_DIR = r"C:\Users\Ananya\Downloads\ADAS_Lane_Project\dataset"

TRAIN_IMG = os.path.join(BASE_DIR, "images", "train")
VAL_IMG = os.path.join(BASE_DIR, "images", "val")
TRAIN_LABEL = os.path.join(BASE_DIR, "labels", "train")
VAL_LABEL = os.path.join(BASE_DIR, "labels", "val")

for folder in [TRAIN_IMG, VAL_IMG, TRAIN_LABEL, VAL_LABEL]:
    os.makedirs(folder, exist_ok=True)

# Find all XML files
xml_files = [
    f for f in os.listdir(XML_DIR)
    if f.lower().endswith(".xml")
]

random.seed(42)
random.shuffle(xml_files)

# 80% training, 20% validation
split = int(0.8 * len(xml_files))

train_files = xml_files[:split]
val_files = xml_files[split:]

# Create image lookup including subfolders
image_files = {}

for root_dir, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files[file] = os.path.join(root_dir, file)


def convert(xml_file, image_output_dir, label_output_dir):

    xml_path = os.path.join(XML_DIR, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = os.path.basename(root.find("filename").text)

    size = root.find("size")

    img_width = float(size.find("width").text)
    img_height = float(size.find("height").text)

    if filename not in image_files:
        print("IMAGE NOT FOUND:", filename)
        return

    # Copy image
    shutil.copy2(
        image_files[filename],
        os.path.join(image_output_dir, filename)
    )

    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(label_output_dir, label_filename)

    with open(label_path, "w") as f:

        for obj in root.findall("object"):

            name = obj.find("name").text

            if name != "road_line":
                continue

            box = obj.find("bndbox")

            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            # Convert to YOLO format
            center_x = ((xmin + xmax) / 2) / img_width
            center_y = ((ymin + ymax) / 2) / img_height

            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(
                f"0 {center_x:.6f} {center_y:.6f} "
                f"{width:.6f} {height:.6f}\n"
            )


print("Converting training data...")

for xml in train_files:
    convert(xml, TRAIN_IMG, TRAIN_LABEL)

print("Converting validation data...")

for xml in val_files:
    convert(xml, VAL_IMG, VAL_LABEL)

print("\n========== DONE ==========")
print("Total XML:", len(xml_files))
print("Training:", len(train_files))
print("Validation:", len(val_files))