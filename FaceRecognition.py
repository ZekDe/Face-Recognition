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

        # dlib'in yÃ¼z dedektÃ¶rÃ¼nÃ¼ ve ÅŸekil tahmin edicisini yÃ¼kleyin

        self.detector = dlib.get_frontal_face_detector()

        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

        print("VideoCapture start")

       

        # Video akÄ±ÅŸÄ±nÄ± baÅŸlatÄ±n (0, laptop kamerasÄ±nÄ± kullanÄ±r)

        #,cv2.CAP_V4L2

        # VideoCapture nesnesini oluÅŸturun

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)



        # FormatÄ± MJPEG olarak ayarlayÄ±n

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))



        # Ã‡Ã¶zÃ¼nÃ¼rlÃ¼ÄŸÃ¼ ayarlayÄ±n

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #1280

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #720

        print("VideoCapture end")



        #print(cv2.getBuildInformation())

        self.lock = threading.Lock()

        self.faces = []

        self.labels = []

        self.face_descriptors = []

        self.face_timestamps = []

        self.personExist = []

        self.face_first_seen = []

        

        self.update_interval = 2  #second

        #ilk first_seen_max_visible_duration sn iÃ§inde invisible_threshold_duration sn'den fazla kamerada gÃ¶rÃ¼nmezse kisiyi kaldir.

        self.invisible_threshold_duration = 5

        self.first_seen_max_visible_duration = 15

        # person_inactivity_duration sn boyunca gÃ¶rÃ¼nmezse kiÅŸi ortamdan ayrÄ±lmÄ±ÅŸtÄ±r

        self.person_inactivity_duration = 30

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

            #self.lock.acquire()

            ret, frame = self.cap.read()

            if not ret:

                print("frame okunamadÄ±!")

                break

            

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            new_faces = self.detector(gray)



            current_time = time.time()

            

            for i, new_face in enumerate(new_faces):

                x, y, w, h = new_face.left(), new_face.top(), new_face.width(), new_face.height()

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                #cv2.putText(frame, f'Person {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)



                new_descriptor = self.getFaceDescriptor(frame, new_face)

                is_new_person = True

                

                # AynÄ± kiÅŸi mi kontrol et 

                # EÄŸer kiÅŸi uzun sÃ¼re iÃ§eride yoksa, tekrar binmiÅŸtir, kiÅŸi sayÄ±sÄ±nÄ± 1 arttÄ±r

                for j, known_descriptor in enumerate(self.face_descriptors):

                    if self.isSamePerson(new_descriptor, known_descriptor):

                        cv2.putText(frame, self.labels[j], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                        is_new_person = False

                        self.face_timestamps[j] = current_time

                        

                        

                        if self.personExist[j] == False:

                            self.total_person_count+=1

                            print(f"Known Person Count:{self.total_person_count}")

                            self.personExist[j] = True

                            self.face_first_seen[j] = current_time

                        break

                # Yeni kiÅŸi ise kayÄ±tlara ekle,  kiÅŸi sayÄ±sÄ±nÄ± 1 arttÄ±r

                if is_new_person:

                    new_label = f'Person {len(self.faces) + 1}'

                    self.faces.append(new_face)

                    self.face_descriptors.append(new_descriptor)

                    self.labels.append(new_label)

                    self.face_timestamps.append(current_time)

                    self.personExist.append(True)

                    self.face_first_seen.append(current_time)

                    self.total_person_count+=1

                    print(f"New Person Count:{self.total_person_count}")

                

                    cv2.putText(frame, new_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            #print(f'Kamera Ã§Ã¶zÃ¼nÃ¼rlÃ¼ÄŸÃ¼: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')

            #print(f'FPS: {self.cap.get(cv2.CAP_PROP_FPS)}')

            #self.lock.release()

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break



            self.checkPersonExistence(current_time)

           # time.sleep(self.update_interval)        

       

        self.cap.release()

        cv2.destroyAllWindows()

        

        

    def checkPersonExistence(self, current_time):

            

            for i, timestamp in enumerate(self.face_timestamps):

                if self.personExist[i] == True and current_time - timestamp > self.person_inactivity_duration:

                    self.personExist[i] = False

                    print(f"{i+1} deleted")

                    

                #ilk 15 iÃ§inde 5 sn'den fazla kamerada gÃ¶rÃ¼nmezse kisiyi kaldir.

                if (self.personExist[i] == True and current_time - self.face_first_seen[i] < self.first_seen_max_visible_duration and

                    current_time - timestamp > self.invisible_threshold_duration):

                    self.personExist[i] = False

                    self.total_person_count-=1

                    print(f"Short visible Count:{self.total_person_count}")



    def run(self):

        thread = threading.Thread(target=self.faceRecognition)

        thread.start()

 





"""                

    def cleanOldFaces(self, current_time):

       # GÃ¶rÃ¼ntÃ¼de gÃ¶rÃ¼nmeyen yÃ¼zleri temizleme

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



       # Listeyi gÃ¼ncelle

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



            # GÃ¶rÃ¼ntÃ¼yÃ¼ griye Ã§evirin

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  



            # YÃ¼zleri algÄ±layÄ±n

            new_faces = self.detector(gray)



            # AlgÄ±lanan yÃ¼zleri Ã§izin ve etiketleyin

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



                    

            # GÃ¶rÃ¼ntÃ¼yÃ¼ gÃ¶sterin

            cv2.imshow('Face Detection', frame)



            # 'q' tuÅŸuna basÄ±ldÄ±ÄŸÄ±nda dÃ¶ngÃ¼yÃ¼ kÄ±rÄ±n

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break



        # Temizlik iÅŸlemleri

        self.cap.release()

        cv2.destroyAllWindows()

        

    def faceRecognition2(self):

        

        flag_face = False

        while self.cap.isOpened():

            ret, frame = self.cap.read()

            if not ret:

                break

    

            # GÃ¶rÃ¼ntÃ¼yÃ¼ griye Ã§evirin

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    

            # YÃ¼zleri algÄ±layÄ±n

            new_faces = self.detector(gray)

    

            flag_face = False

            # AlgÄ±lanan yÃ¼zleri Ã§izin ve etiketleyin

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

                    

            # GÃ¶rÃ¼ntÃ¼yÃ¼ gÃ¶sterin

            cv2.imshow('Face Detection', frame)

    

            # 'q' tuÅŸuna basÄ±ldÄ±ÄŸÄ±nda dÃ¶ngÃ¼yÃ¼ kÄ±rÄ±n

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

    

        # Temizlik iÅŸlemleri

        self.cap.release()

        cv2.destroyAllWindows()

"""

