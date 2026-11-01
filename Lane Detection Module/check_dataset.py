import os
import xml.etree.ElementTree as ET

IMAGE_DIR = r"C:\Users\Ananya\Downloads\LANE DETECTION\LANE DETECTION\road_line_images\road_line_images"
XML_DIR = r"C:\Users\Ananya\Downloads\LANE DETECTION\LANE DETECTION\road_line_annotation\road_line_annotation"

xml_files = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]

# Create a list of ALL JPG files including subfolders
image_files = {}

for root_dir, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files[file] = os.path.join(root_dir, file)

matched = 0
missing = 0

for xml_file in xml_files:

    xml_path = os.path.join(XML_DIR, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = os.path.basename(root.find("filename").text)

    if filename in image_files:
        matched += 1
    else:
        missing += 1
        print("MISSING IMAGE:", filename)

print("\n========== RESULT ==========")
print("XML files:", len(xml_files))
print("Matched images:", matched)
print("Missing images:", missing)