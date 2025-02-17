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

    cap = cv2.VideoCapture(0)

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

            min_box_height = frame_height * 0.10  
            max_box_height = frame_height * 0.50  

            print(f"\nFrame Height: {frame_height}, Box Height: {box_height}")
            print(f"Min Box Height: {min_box_height}, Max Box Height: {max_box_height}")

            if text_detected:
                current_time = time.time()

                if current_time - last_distance_feedback_time > 1.5:
                    if box_height == 0:
                        print("No box detected, skipping distance check.")
                    elif box_height < min_box_height:
                        print("Too Small → Move Closer")
                        convert_text_to_audio("Move closer to the camera.")
                    elif box_height > max_box_height:
                        print("Too Large → Move Farther")
                        convert_text_to_audio("Move farther from the camera.")
                    else:
                        print("Box size is good.")

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
                print(f"\n🔍 Box Height: {box_height} | Min: {min_box_height} | Max: {max_box_height}")

                current_time = time.time()
                if current_time - last_distance_feedback_time > 1.5:
                    if box_height == 0:
                        print("No box detected, skipping distance check.")
                    elif box_height < min_box_height:
                        print("Too Small → Move Closer")
                        convert_text_to_audio("Move closer to the camera.")
                    elif box_height > max_box_height:
                        print("Too Large → Move Farther")
                        convert_text_to_audio("Move farther from the camera.")
                    else:
                        print("Box size is good.")

                    last_distance_feedback_time = current_time  

            if text_detected and confidence >= ACCURACY_THRESHOLD:
                batch_results.append((detected_text, confidence))
                print(f"Detected: {detected_text} (Confidence: {confidence:.2f}%)")

            if time.time() - start_time >= SCAN_DURATION:
                scanning = False
                print("Scanning completed.")

                if batch_results:
                    best_result = max(batch_results, key=lambda x: x[1])
                    best_text, best_confidence = best_result
                    print(f"🏆 Best result: {best_text} (Confidence: {best_confidence:.2f}%)")
                    convert_text_to_audio(best_text)
                else:
                    print("⚠️ No high-confidence text detected.")

        if key == ord('q'):
            print("🚪 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
