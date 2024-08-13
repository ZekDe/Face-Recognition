# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:18:35 2024

@author: Duatepe
"""
"""
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
"""
#import faceRecognition
import time
import threading
from FaceRecognition import FaceRecognition 
#import keyboard



def getFaces(obj):
    while True:
        if obj:
            print("Algılanan Yüzler:")
            for i in range(len(obj.labels)):
                print(f'{obj.labels[i]}: face={obj.faces[i]}')
            print(f'Toplam Kişi Sayısı: {obj.total_person_count}')
        
        time.sleep(1)

#        if keyboard.is_pressed('q'):
#            break
        

def main():
    face_recognition = FaceRecognition()
    face_recognition.run()
    
    thread = threading.Thread(target=getFaces, args=(face_recognition,))
    thread.start()
    
    

    
if __name__ == "__main__":
    main()
    
   
    
