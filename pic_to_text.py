import cv2
import pytesseract
import numpy as np

#perform OCR on frames
def perform_ocr(frame):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to extract data with confidence value
    data = pytesseract.image_to_data(gray_frame, lang='eng', output_type= \
                                    pytesseract.Output.DICT)
    
    # Extract text and calculate average confidence
    text = "\n".join([data['text'][i] for i in range(len(data['text'])) \
                    if data['text'][i].strip()])

    confidences = [int(str(data['conf'][i])) for i in range(len(data['conf'])) \
                    if str(data['conf'][i]).isdigit()]

    average_confidence = np.mean(confidences) if confidences else 0

    

    return text, average_confidence
