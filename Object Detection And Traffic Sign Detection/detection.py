from ultralytics import YOLO
import cv2
import os
import glob

# ============================================================
# LOAD YOLOv11n
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# INPUT AND OUTPUT FOLDERS
# ============================================================

input_folder = "input"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# GET ALL IMAGES FROM INPUT FOLDER
# ============================================================

image_files = []

for extension in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    image_files.extend(glob.glob(os.path.join(input_folder, extension)))


if len(image_files) == 0:
    print("No images found in input folder.")
    exit()


print("======================================")
print("ADAS OBJECT DETECTION")
print("Images found:", len(image_files))
print("======================================")


# ============================================================
# CHECK WHETHER TWO LABEL BOXES OVERLAP
# ============================================================

def is_overlapping(box1, box2):

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    if x1 >= a2 or a1 >= x2:
        return False

    if y1 >= b2 or b1 >= y2:
        return False

    return True


# ============================================================
# FIND FREE POSITION FOR LABEL
# ============================================================

def find_free_position(
    x,
    y,
    label_width,
    label_height,
    image_width,
    image_height,
    occupied_boxes,
    title_height,
    bottom_panel_height
):

    positions = [

        # Above
        (x, y - label_height - 5),

        # Below
        (x, y + label_height + 5),

        # Right
        (x + label_width + 5, y),

        # Left
        (x - label_width - 5, y),

        # Further above
        (x, y - (label_height * 2) - 10),

        # Further below
        (x, y + (label_height * 2) + 10),

        # Further right
        (x + label_width + 15, y),

        # Further left
        (x - label_width - 15, y)
    ]


    for pos_x, pos_y in positions:

        # Keep inside image
        pos_x = max(
            5,
            min(
                pos_x,
                image_width - label_width - 5
            )
        )

        pos_y = max(
            title_height + 5,
            pos_y
        )


        # Don't place inside bottom panel
        if pos_y + label_height > image_height - bottom_panel_height:
            continue


        candidate = (
            pos_x,
            pos_y,
            pos_x + label_width,
            pos_y + label_height
        )


        overlap = False

        for occupied in occupied_boxes:

            if is_overlapping(candidate, occupied):

                overlap = True
                break


        if not overlap:

            return pos_x, pos_y


    # ========================================================
    # FALLBACK SEARCH
    # ========================================================

    pos_x = max(
        5,
        min(
            x,
            image_width - label_width - 5
        )
    )

    pos_y = title_height + 10


    while pos_y + label_height < image_height - bottom_panel_height:

        candidate = (
            pos_x,
            pos_y,
            pos_x + label_width,
            pos_y + label_height
        )

        overlap = False

        for occupied in occupied_boxes:

            if is_overlapping(candidate, occupied):

                overlap = True
                break


        if not overlap:

            return pos_x, pos_y


        pos_y += label_height + 10


    return pos_x, title_height + 5


# ============================================================
# PROCESS EVERY IMAGE
# ============================================================

for image_index, image_path in enumerate(image_files, start=1):

    print()
    print("======================================")
    print("Processing image:", image_index)
    print("File:", image_path)
    print("======================================")


    # ========================================================
    # RUN YOLO
    # ========================================================

    results = model(
        image_path,
        conf=0.01
    )


    for result in results:

        # Original image
        image = result.orig_img.copy()

        height, width = image.shape[:2]


        print("Image Width:", width)
        print("Image Height:", height)
        print("--------------------------------------")


        # ====================================================
        # UI SETTINGS
        # ====================================================

        title_height = 70
        bottom_panel_height = 70


        # ====================================================
        # TITLE
        # ====================================================

        cv2.rectangle(
            image,
            (0, 0),
            (width, title_height),
            (40, 40, 40),
            -1
        )


        title = "AI-POWERED ADAS - OBJECT DETECTION"


        title_size = cv2.getTextSize(
            title,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            3
        )[0]


        title_x = max(
            10,
            (width - title_size[0]) // 2
        )


        cv2.putText(
            image,
            title,
            (title_x, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )


        # ====================================================
        # LABEL STORAGE
        # ====================================================

        occupied_boxes = []


        # Count only visible objects
        object_count = 0


        # ====================================================
        # DETECTION LOOP
        # ====================================================

        for box in result.boxes:


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(box.conf[0])


            # ------------------------------------------------
            # HIDE BELOW 0.30
            # ------------------------------------------------

            if confidence < 0.30:

                continue


            # Count visible objects only
            object_count += 1


            # ------------------------------------------------
            # CLASS ID
            # ------------------------------------------------

            class_id = int(box.cls[0])


            # ------------------------------------------------
            # CLASS NAME
            # ------------------------------------------------

            class_name = model.names[class_id]


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = box.xyxy[0].tolist()


            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            # ------------------------------------------------
            # CENTER POINT
            # ------------------------------------------------

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # =================================================
            # DETERMINE LEFT / CENTER / RIGHT
            # =================================================

            if center_x < width / 3:

                position = "LEFT"

            elif center_x < 2 * width / 3:

                position = "CENTER"

            else:

                position = "RIGHT"


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                4
            )


            # =================================================
            # DRAW CENTER POINT
            # =================================================

            cv2.circle(
                image,
                (center_x, center_y),
                8,
                (0, 0, 255),
                -1
            )


            # =================================================
            # CREATE LABEL
            # =================================================

            label = (
                f"{class_name.upper()}  |  "
                f"{confidence:.2f}  |  "
                f"{position}"
            )


            # =================================================
            # LABEL SIZE
            # =================================================

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.65

            thickness = 2


            text_size = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )[0]


            label_width = text_size[0] + 20

            label_height = 42


            # =================================================
            # FIND NON-OVERLAPPING LABEL POSITION
            # =================================================

            label_x, label_y = find_free_position(
                x1,
                y1,
                label_width,
                label_height,
                width,
                height,
                occupied_boxes,
                title_height,
                bottom_panel_height
            )


            # =================================================
            # SAVE LABEL POSITION
            # =================================================

            occupied_boxes.append(
                (
                    label_x,
                    label_y,
                    label_x + label_width,
                    label_y + label_height
                )
            )


            # =================================================
            # POSITION COLOUR
            # =================================================

            if position == "LEFT":

                box_color = (255, 100, 0)

            elif position == "CENTER":

                box_color = (0, 140, 255)

            else:

                box_color = (0, 180, 0)


            # =================================================
            # DRAW LABEL BOX
            # =================================================

            cv2.rectangle(
                image,
                (label_x, label_y),
                (
                    label_x + label_width,
                    label_y + label_height
                ),
                box_color,
                -1
            )


            # =================================================
            # LABEL BORDER
            # =================================================

            cv2.rectangle(
                image,
                (label_x, label_y),
                (
                    label_x + label_width,
                    label_y + label_height
                ),
                (255, 255, 255),
                2
            )


            # =================================================
            # LABEL TEXT
            # =================================================

            cv2.putText(
                image,
                label,
                (label_x + 10, label_y + 29),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )


            # =================================================
            # TERMINAL OUTPUT
            # =================================================

            print("Object:", class_name)
            print("Confidence:", round(confidence, 2))
            print("Center:", center_x, center_y)
            print("Position:", position)
            print("--------------------------------------")


        # ====================================================
        # BOTTOM PANEL
        # ====================================================

        cv2.rectangle(
            image,
            (0, height - bottom_panel_height),
            (width, height),
            (35, 35, 35),
            -1
        )


        # ====================================================
        # OBJECT COUNT
        # ====================================================

        cv2.putText(
            image,
            f"OBJECTS DETECTED: {object_count}",
            (25, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # MODEL NAME
        # ====================================================

        model_text = "MODEL: YOLOv11n"


        model_size = cv2.getTextSize(
            model_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )[0]


        cv2.putText(
            image,
            model_text,
            (
                width - model_size[0] - 25,
                height - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # SAVE OUTPUT
        # ====================================================

        filename = os.path.basename(image_path)

        name, extension = os.path.splitext(filename)

        output_path = os.path.join(
            output_folder,
            f"{name}_result{extension}"
        )


        cv2.imwrite(
            output_path,
            image
        )


        print()
        print("Saved:", output_path)


        # ====================================================
        # SHOW RESULT
        # ====================================================

        cv2.imshow(
            "ADAS Detection System",
            image
        )


        cv2.waitKey(0)

        cv2.destroyAllWindows()


# ============================================================
# FINISHED
# ============================================================

print()
print("======================================")
print("ALL IMAGES PROCESSED SUCCESSFULLY")
print("======================================")