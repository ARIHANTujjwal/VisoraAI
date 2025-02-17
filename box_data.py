import cv2
import pytesseract
import numpy as np

def detect_text_and_draw_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)

    text_detected = False
    detected_text = ""
    boxes = []
    confidences = []
    word_boxes = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = data['conf'][i] if isinstance(data['conf'][i], int) else 0
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

        if conf > 60 and text:
            text_detected = True
            detected_text += text + " "
            boxes.append((x, y, x + w, y + h))
            confidences.append(conf)
            word_boxes.append((x, y, w, h))

    # Draw green boxes around words and document
    if text_detected and boxes:
        for box in boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Compute average confidence
    average_confidence = np.mean(confidences) if confidences else 0
    box_height = max([box[3] - box[1] for box in boxes]) if boxes else 0

    return frame, text_detected, detected_text.strip(), average_confidence, box_height, word_boxes
