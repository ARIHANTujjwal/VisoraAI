from gtts import gTTS
import os

#convert text to audio
def convert_text_to_audio(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("current_text.mp3")
        os.system("mpg321 current_text.mp3")  
    except Exception as e:
        print(f"Error in text-to-audio conversion: {e}")
