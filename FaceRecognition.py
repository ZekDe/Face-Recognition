# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:01:22 2024

@author: Duatepe

@Project: Face recognition demo

"""

import dlib
import cv2

import threading
import numpy as np
import time

class FaceRecognition:
    def __init__(self):
        # dlib'in yüz dedektörünü ve şekil tahmin edicisini yükleyin
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
       # self. face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Video akışını başlatın (0, laptop kamerasını kullanır)
        self.cap = cv2.VideoCapture(0)

        # Çözünürlüğü ayarlayın
        self.lock = threading.Lock()
        self.faces = []
        self.labels = []
        self.face_descriptors = []
        self.face_timestamps = []
        self.personExist = []
        
        self.update_interval = 2  #second
        self.person_inactivity_duration = 5
        self.total_person_count = 0
        
        
    def getFaceDescriptor(self, frame, face):
     #   if frame is None or face is None:
     #      return np.array([])
        shape = self.shape_predictor(frame, face)
        return np.array(self.face_rec_model.compute_face_descriptor(frame, shape))

    def isSamePerson(self, descriptor1, descriptor2, threshold=0.6):
        return np.linalg.norm(descriptor1 - descriptor2) < threshold
    

    def faceRecognition(self):
        while self.cap.isOpened():
            self.lock.acquire()
            ret, frame = self.cap.read()
            if not ret:
                print("Kamera frame'i okunamadı")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_faces = self.detector(gray)
            #new_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            #if not new_faces:
             #   self.lock.release()
              #  continue
            current_time = time.time()
            self.checkPersonExistence(current_time)
            for i, new_face in enumerate(new_faces):
                x, y, w, h = new_face.left(), new_face.top(), new_face.width(), new_face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                #cv2.putText(frame, f'Person {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                new_descriptor = self.getFaceDescriptor(frame, new_face)
                is_new_person = True
                
                # Aynı kişi mi kontrol et 
                # Eğer kişi uzun süre içeride yoksa, tekrar binmiştir, kişi sayısını 1 arttır
                for j, known_descriptor in enumerate(self.face_descriptors):
                    if self.isSamePerson(new_descriptor, known_descriptor):
                        cv2.putText(frame, self.labels[j], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        is_new_person = False
                        self.face_timestamps[j] = current_time
                        
                        if self.personExist[j] == False:
                            self.total_person_count+=1
                            self.personExist[j] = True
                        break
                # Yeni kişi ise kayıtlara ekle,  kişi sayısını 1 arttır
                if is_new_person:
                    new_label = f'Person {len(self.faces) + 1}'
                    self.faces.append(new_face)
                    self.face_descriptors.append(new_descriptor)
                    self.labels.append(new_label)
                    self.face_timestamps.append(current_time)
                    self.personExist.append(True)
                    self.total_person_count+=1
                    cv2.putText(frame, new_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            
            
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.lock.release()
               # time.sleep(self.update_interval)    
        self.cap.release()
        cv2.destroyAllWindows()
            
        
    def checkPersonExistence(self, current_time):
        for i, timestamp in enumerate(self.face_timestamps):
            if current_time - timestamp > self.person_inactivity_duration:
                self.personExist[i] = False
    
    def run(self):
        thread = threading.Thread(target=self.faceRecognition)
        thread.start()
 


"""                
    def cleanOldFaces(self, current_time):
       # Görüntüde görünmeyen yüzleri temizleme
       active_faces = []
       active_descriptors = []
       active_labels = []
       active_timestamps = []

       for i, timestamp in enumerate(self.face_timestamps):
           if current_time - timestamp < self.person_inactivity_duration:
               active_faces.append(self.faces[i])
               active_descriptors.append(self.face_descriptors[i])
               active_labels.append(self.labels[i])
               active_timestamps.append(self.face_timestamps[i])

       # Listeyi güncelle
       self.faces = active_faces
       self.face_descriptors = active_descriptors
       self.labels = active_labels
       self.face_timestamps = active_timestamps
        
        

        
        
       
 
    def faceRecognition1(self):
        
        flag_face = False
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Görüntüyü griye çevirin
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

            # Yüzleri algılayın
            new_faces = self.detector(gray)

            # Algılanan yüzleri çizin ve etiketleyin
            for i, new_face in enumerate(new_faces):
                x, y, w, h = new_face.left(), new_face.top(), new_face.width(), new_face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f'Person {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
                new_descriptor = self.get_face_descriptor(frame, new_face)
                is_new_person = True
                
                for j, known_descriptor in enumerate(self.face_descriptors):
                    if self.is_same_person(new_descriptor, known_descriptor):
                        is_new_person = False
                        break
                    
                if is_new_person:
                        self.faces.append(new_face)
                        self.face_descriptors.append(new_descriptor)
                        self.labels.append(f'Person {i+1}')

                    
            # Görüntüyü gösterin
            cv2.imshow('Face Detection', frame)

            # 'q' tuşuna basıldığında döngüyü kırın
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Temizlik işlemleri
        self.cap.release()
        cv2.destroyAllWindows()
        
    def faceRecognition2(self):
        
        flag_face = False
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
    
            # Görüntüyü griye çevirin
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
            # Yüzleri algılayın
            new_faces = self.detector(gray)
    
            flag_face = False
            # Algılanan yüzleri çizin ve etiketleyin
            for i, new_face in enumerate(new_faces):
                x, y, w, h = new_face.left(), new_face.top(), new_face.width(), new_face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f'Person {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
                for old_face in self.faces:
                    if new_face == old_face:
                        flag_face = True
                        
                if flag_face == False:
                    self.faces.append(new_face)
                    self.labels.append(f'Person {i+1}')
                    
            # Görüntüyü gösterin
            cv2.imshow('Face Detection', frame)
    
            # 'q' tuşuna basıldığında döngüyü kırın
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        # Temizlik işlemleri
        self.cap.release()
        cv2.destroyAllWindows()
"""