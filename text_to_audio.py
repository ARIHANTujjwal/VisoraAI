from gtts import gTTS
import os
#import time

#convert text to audio
def convert_text_to_audio(text):
    try:
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        #fname = f"outputs/audio_{timestr}.mp3"
        fname = "outputs/current_text.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(fname)
        #resp_name = "mpg321" + fname
        os.system("mpg321 outputs/current_text.mp3")  
    except Exception as e:
        print(f"Error in text-to-audio conversion: {e}")
