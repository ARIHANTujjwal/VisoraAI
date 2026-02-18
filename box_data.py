import cv2
import numpy as np
import pytesseract

def detect_text_and_draw_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    #gray is 2d array of hight and width
    '''
    output type is a dictionay that stores:
    {
  'level': [...],
  'page_num': [...],
  'block_num': [...],
  'par_num': [...],
  'line_num': [...],
  'word_num': [...],
  'left': [...],
  'top': [...],
  'width': [...],
  'height': [...],
  'conf': [...],
  'text': [...]
    }
    '''
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)

    text_detected = False
    detected_text = ""
    boxes = []
    confidences = []

    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0  # creates box around all text(important sections)

    for i in range(len(data['text'])):
        #removes empty spaces
        text = data['text'][i].strip()
        #conf = data['conf'][i] if isinstance(data['conf'][i], int) else 0
        # ensure conf is always an int
        if isinstance(data['conf'][i], int):
            conf = data['conf'][i]
        else:
            conf = 0

        # provides dimensions of box around every text 
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

        if conf > 60 and text:
            text_detected = True
            detected_text += text + " "
            boxes.append((x, y, x + w, y + h))
            #can be used to show codes accuracy
            confidences.append(conf)

            # Update box dimentions around whole text section
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

    # Draw small boxes around individual words, colors each box green
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for words

    # Draw a large box around the entire collection of words
    if text_detected:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)  # Blue box for the whole text block

    # Compute average confidence
    average_confidence = np.mean(confidences) if confidences else 0

    return frame, text_detected, detected_text.strip(), average_confidence, y_max - y_min, boxes


# note: .strip() removes unwanted whitespace from both ends of a string:
