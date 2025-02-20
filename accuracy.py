import cv2
import os
from Levenshtein import distance
from pic_to_text import perform_ocr  # Import OCR function from VisoraAI

def clean_text(text):
    """Clean text by removing unwanted characters like newlines and extra spaces."""
    return " ".join(text.replace("\n", " ").split())

def calculate_cer(ground_truth, detected_text):
    """Calculate Character Error Rate (CER) using Levenshtein distance."""
    cleaned_detected_text = clean_text(detected_text)
    cleaned_ground_truth = clean_text(ground_truth)
    
    if not cleaned_ground_truth:
        return 100 if cleaned_detected_text else 0  
    return (distance(cleaned_ground_truth, cleaned_detected_text) / len(cleaned_ground_truth)) * 100

def evaluate_ocr(image_path, ground_truth):
    """Run OCR on an image and compare it to ground truth."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Ensure we are extracting only the text part from the tuple
    detected_text, _ = perform_ocr(image)
    
    cer = calculate_cer(ground_truth, detected_text)

    print(f"\nImage: {image_path}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Detected: {detected_text}")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    
    return cer

def run_accuracy_test():
    """Evaluate multiple images for overall OCR accuracy."""
    test_data = [
        ("static_files/example1.jpg", "Hi I am visora, I am very cool."),
        ("static_files/example2.jpg", "This is a very interesting book."),
        # Add more test cases here...
    ]

    total_cer = 0
    valid_tests = 0

    for img_path, ground_truth in test_data:
        if os.path.exists(img_path):
            cer = evaluate_ocr(img_path, ground_truth)
            if cer is not None:
                total_cer += cer
                valid_tests += 1
        else:
            print(f"Warning: {img_path} not found. Skipping...")

    if valid_tests > 0:
        avg_cer = total_cer / valid_tests
        print(f"\nOverall OCR Accuracy: {100 - avg_cer:.2f}%")
    else:
        print("No valid test cases were run.")

if __name__ == "__main__":
    run_accuracy_test()
