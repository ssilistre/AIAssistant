import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import threading
import time
import speech_recognition as sr
#import func
import sys

class ImageCaptioning:

    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float32).to("cpu")
        self.last_frame = None
        self.caption = ""

    def listen_to_speech(self):
        recognizer = sr.Recognizer()
        while True:  # Sürekli dinleme için döngü
            with sr.Microphone() as source:
                print("Dinleniyor...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                try:
                    text = recognizer.recognize_google(audio, language='tr-TR')
                    print("Algılanan: " + text)
                    self.handle_speech_command(text)
                except sr.UnknownValueError:
                    print("Anlaşılamadı")
                except sr.RequestError:
                    print("Servis hatası")


    def handle_speech_command(self, command):
        if "merhaba" in command.lower():
            print("Merhaba, nasıl yardımcı olabilirim?")
        elif "kamera durumu" in command.lower():
            print("Kamera şu anda aktif.")
        elif "yardım et" in command.lower():
            print("Kullanabileceğiniz komutlar: merhaba, kamera durumu, yardım et, programı kapat")
        elif "programı kapat" in command.lower():
            print("Program kapatılıyor...")
            sys.exit()
        elif "chrome aç" in command.lower():
            print("Chrome başlatılıyor...")
            #func.open_chrome()
        elif "ne görüyorsun" in command.lower():
            #cv2.imwrite('son_cerceve.jpg', self.last_frame)
            #print("Çerçeve kaydedildi.")
            #func.chat_with_openai("Gördüğü Şey:"+self.caption)
        else:
            print("Tanınmayan komut: " + command)

    def process_image(self):
        while True:
            if self.last_frame is not None:
                inputs = self.processor(self.last_frame, return_tensors="pt").to("cpu", torch.float32)
                out = self.model.generate(**inputs, max_new_tokens=50) 
                self.caption = self.processor.decode(out[0], skip_special_tokens=True)
                print("Caption: " + self.caption)
                self.last_frame = None
            time.sleep(5)  # Anlamlandırma sıklığını azalt

    def capture_and_process_image(self):
        cap = cv2.VideoCapture(0)
        threading.Thread(target=self.process_image, daemon=True).start()
        threading.Thread(target=self.listen_to_speech, daemon=True).start()

        #Yüz tanıma ekleyelim.
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))

            # Yüz tanıma işlemi
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Her yüz için kırmızı bir dikdörtgen çiz
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            self.last_frame = frame
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    image_captioning = ImageCaptioning()
    image_captioning.capture_and_process_image()
