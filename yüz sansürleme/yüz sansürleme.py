import cv2
import numpy as np

# Haarcascade yüz tanıma modelini yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Maskeleme için bir örnek maske görüntüsü yükle
mask_img = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Gri tonlamalı görüntüyü elde et
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Yüzü kapsayan bir dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Yüzü kapsayan bölgeyi maskele
        mask_resized = cv2.resize(mask_img, (w, h))
        mask_inv = cv2.bitwise_not(mask_resized[:,:,3])
        mask = mask_resized[:,:,0:3]
        roi = frame[y:y+h, x:x+w]
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        roi_fg = cv2.bitwise_and(mask, mask, mask=mask_resized[:,:,3])
        masked_face = cv2.add(roi_bg, roi_fg)

        # Maskeleme sonucunu orijinal kareye yerleştir
        frame[y:y+h, x:x+w] = masked_face

    # Görüntüyü göster
    cv2.imshow('Face Masking', frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

# Kamerayı kapat
cap.release()
cv2.destroyAllWindows()
