import cv2
import numpy as np

def lite_detect_text_boxes(frame_bgr):
    """
    OpenCV-only text detector using masking + contours.
    Returns:
      processed_frame (with green word boxes + blue overall box),
      word_boxes: list of (x1,y1,x2,y2)
    """
    img = frame_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binary mask (adaptive works better for uneven lighting)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 9
    )

    # Ensure black text on white background (CRNN-friendly)
    # If the image is mostly black, invert it
    if np.mean(bw) < 127:
        bw = 255 - bw

    # Connect characters within words (horizontal emphasis)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    cnts, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    word_boxes = []

    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)

        # Filter noise
        if ww < 18 or hh < 10:
            continue
        if ww > 0.98 * w or hh > 0.98 * h:
            continue

        area = ww * hh
        if area < 180:
            continue

        aspect = ww / max(1, hh)
        # word-ish regions: usually wider than tall
        if aspect < 1.0:
            continue

        word_boxes.append((x, y, x + ww, y + hh))

    # Sort top-to-bottom, left-to-right (basic reading order)
    word_boxes.sort(key=lambda b: (b[1], b[0]))

    # Draw word boxes + overall block box
    x_min, y_min, x_max, y_max = 10**9, 10**9, 0, 0
    for (x1, y1, x2, y2) in word_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x_min = min(x_min, x1)
        y_min = min(y_min, y1)
        x_max = max(x_max, x2)
        y_max = max(y_max, y2)

    if word_boxes:
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    return img, word_boxes
