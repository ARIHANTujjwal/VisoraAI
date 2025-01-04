import cv2
import time
from pic_to_text import perform_ocr
from text_to_audio import convert_text_to_audio

def main():
    # Define the accuracy threshold and timer duration
    ACCURACY_THRESHOLD = 85
    SCAN_DURATION = 5

    # Initialize variables for scanning and storing results
    scanning = False
    start_time = None
    batch_results = []


    cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return


    print("Press 's' to start scanning for 5 seconds.")
    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Live Webcam Feed", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not scanning:
            # Start scanning
            scanning = True
            start_time = time.time()
            batch_results = []
            print("Scanning started...")

        if scanning:
            # Perform OCR on the current frame
            text, confidence = perform_ocr(frame)

            # Store results if confidence exceeds the threshold
            if text.strip() and confidence >= ACCURACY_THRESHOLD:
                batch_results.append((text, confidence))
                print(f"Detected: {text} (Confidence: {confidence:.2f}%)")

            # Check if the scanning interval is over
            if time.time() - start_time >= SCAN_DURATION:
                scanning = False
                print("Scanning completed.")

                # Process the batch results
                if batch_results:
                    # Find the text with the highest confidence
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
