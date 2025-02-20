# VisoraAI: Assistive OCR for Blind Users

VisoraAI is a Python-based project designed to help visually impaired users read text from their surroundings. Using a webcam for real-time text recognition, it identifies text with high accuracy and converts it to speech.

---

## Features

- **Real-Time OCR**: Extracts text from the live webcam feed.
- **High Confidence Processing**: Processes only high-confidence text batches.
- **Text-to-Speech Conversion**: Converts the extracted text to English audio.
- **User-Friendly Trigger**: Scans for text and reads aloud upon a manual trigger.
- **Batch-Based Accuracy**: Processes and speaks only the most accurate text detected in 5-second intervals.

---

## Project Structure

### Files

1. **`pic_to_text.py`**  
   Handles real-time text extraction from webcam frames using Tesseract OCR.

2. **`text_to_audio.py`**  
   Converts the recognized text into English audio using Google Text-to-Speech (gTTS).

3. **`visora_ai.py`**  
   The main program that orchestrates the scanning, text recognition, and speech generation.

---

## Installation

### Prerequisites
- Python 3.8+
- Poetry package manager
- Webcam (external or built-in)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/VisoraAI.git
   cd VisoraAI

### How to run:
poetry run python3 visora_ai.py  

### Git push:
git add --all
git commit -m "(change details)"
git push -u origin main
