import cv2
import time
import sys
import numpy as np
sys.path.append('/Users/arihantujjwal/Documents/VisoraAI')

from pic_to_text import perform_ocr
from text_to_audio import convert_text_to_audio
from box_data import detect_text_and_draw_box

def main():
    ACCURACY_THRESHOLD = 75  
    SCAN_DURATION = 5
    scanning = False
    start_time = None
    batch_results = []
    last_distance_feedback_time = 0  

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to start scanning for 5 seconds.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        key = cv2.waitKey(1) & 0xFF

        if not scanning:
            cv2.imshow("Live Webcam Feed", frame)

        if key == ord('s') and not scanning:
            processed_frame, text_detected, detected_text, confidence, box_height, word_boxes = detect_text_and_draw_box(frame)

            frame_height = frame.shape[0]
            min_box_height = frame_height * 0.15  
            medium_box_height = frame_height * 0.30  
            
            print(f"Frame Height: {frame_height}, Box Height: {box_height}")
            print(f"Min Box Height: {min_box_height}, Medium Box Height: {medium_box_height}")

            if text_detected:
                current_time = time.time()

                black_pixels_ratio = np.count_nonzero(processed_frame == 0) / frame.size
                overlap_ratio = np.count_nonzero((processed_frame == 255) & (frame == 255)) / frame.size

                print(f"Black Pixels Ratio: {black_pixels_ratio:.2f}, Overlap Ratio: {overlap_ratio:.2f}")

                if current_time - last_distance_feedback_time > 1.5:
                    if box_height == 0 or black_pixels_ratio < 0.015:  
                        print("Too Dark or No Text → Move Closer and Reclick Scan Button")
                        convert_text_to_audio("Move closer to the camera and press the scan button again.")
                        continue  
                    elif box_height < medium_box_height and black_pixels_ratio < 0.25 and overlap_ratio < 0.25:
                        print("Text Too Small and Limited Overlap → Move Closer and Reclick Scan Button")
                        convert_text_to_audio("Move closer to the camera and press the scan button again.")
                        continue  
                    else:
                        print("Book is clear. Scanning can start.")
                        convert_text_to_audio("Book is clear. Scanning now.")

                    last_distance_feedback_time = current_time  

                if confidence >= 60 and len(detected_text.split()) > 3:
                    convert_text_to_audio("Text detected. Analyzing text...")

            scanning = True
            start_time = time.time()
            batch_results = []
            print("Scanning started...")

        if scanning:
            processed_frame, text_detected, detected_text, confidence, box_height, word_boxes = detect_text_and_draw_box(frame)
            
            if text_detected:
                mask = np.zeros_like(processed_frame)
                for (x, y, w, h) in word_boxes:
                    mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
                processed_frame = mask

            cv2.imshow("Live Webcam Feed", processed_frame)

            if text_detected and confidence >= ACCURACY_THRESHOLD:
                batch_results.append((detected_text, confidence))
                print(f"Detected: {detected_text} (Confidence: {confidence:.2f}%)")

            if time.time() - start_time >= SCAN_DURATION:
                scanning = False
                print("Scanning completed.")

                if batch_results:
                    best_result = max(batch_results, key=lambda x: x[1])
                    best_text, best_confidence = best_result
                    print(f"Best result: {best_text} (Confidence: {best_confidence:.2f}%)")
                    convert_text_to_audio(best_text)
                else:
                    print("No high-confidence text detected.")

        if key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
