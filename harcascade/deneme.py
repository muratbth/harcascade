import cv2
import numpy as np
from os import path


cap = cv2.VideoCapture(0)

# Yüz tanıma sınıflandırıcısını yükle
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

isim = input("Adınız ve Soyadınız ? : ")

# Yüz örneği sayısı
sayac = 30

# Yüz listesi oluştur
yuz_listesi = []

while True:
    ret, kare = cap.read()

    if ret:
       
        gray = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)

        # Yüzleri algıla
        yuzler = classifier.detectMultiScale(gray)

        alanlar = []
        for yuz in yuzler:
            x, y, w, h = yuz
            alanlar.append((w * h, yuz))

        if len(yuzler) > 0:
            yuz = max(alanlar)[1]
            x, y, w, h = yuz

            # Yüzü kırp ve yeniden boyutlandır
            yuz_img = gray[y:y+h, x:x+w]
            yuz_img = cv2.resize(yuz_img, (100, 100))
            yuz_dizisi = yuz_img.flatten()
            yuz_listesi.append(yuz_dizisi)
            sayac -= 1
            print("Yüklenen yüz sayısı:", 30 - sayac)
            if sayac <= 0:
                break

            # Yüz görüntüsünü ekranda göster
            cv2.imshow("video", yuz_img)

    
    tus = cv2.waitKey(1)
    if tus & 0xff == ord('q'):
        break

# Yüz verilerini ve isimleri bir araya getir
X = np.array(yuz_listesi)
y = np.full((len(X), 1), isim)

veri = np.hstack([y, X])

print(veri.shape)
print(veri.dtype)

# Kamerayı kapat ve pencereleri yok et
cap.release()
cv2.destroyAllWindows()

# Yüz verilerini kaydet
if path.exists("face_tdata.npy"):
    yuz_verisi = np.load("face_tdata.npy")
    yuz_verisi = np.vstack([yuz_verisi, veri])
    np.save("face_tdata.npy", yuz_verisi)
else:
    np.save("face_tdata.npy", veri)
