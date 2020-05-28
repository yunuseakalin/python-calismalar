import cv2
import numpy as np
import os

kamera = cv2.VideoCapture(0)
kernel = np.ones((5, 5), dtype=np.float32) / 225
def ResimFarkBul(resim1,resim2):

    resim2 = cv2.resize(resim2, (resim1.shape[1],resim1.shape[0]))
    fark_resim = cv2.absdiff(resim1,resim2)
    fark_sayi = cv2.countNonZero(fark_resim)
    cv2.imshow("fark",fark_resim)
    return fark_sayi

def VeriYukle():
    
    veri_isimler = []
    veri_resimler = []
    Dosyalar = os.listdir("veri")

    for Dosya in Dosyalar:
        isim = Dosya.replace(".jpg", "")
        veri_isimler.append(isim)
        veri_resimler.append(cv2.imread("veri/"+Dosya,0))
    return veri_isimler,veri_resimler

veri_isimler, veri_resimler = VeriYukle()
print(veri_isimler)
veri = VeriYukle()

def Siniflandir(resim,veri_isimler,veri_resimler):
    min_index = 0
    min_deger = ResimFarkBul(resim,veri_resimler[0])
    for t in range(len(veri_isimler)):
        fark_deger = ResimFarkBul(resim,veri_resimler[t])
        if (fark_deger < min_deger):
            min_deger = fark_deger
            min_index = t
    return veri_isimler[min_index]

while True:
    ret,kare = kamera.read()
    kesilmis_kare = kare[0:250,0:250]

    kesilmis_kare_gri = cv2.cvtColor(kesilmis_kare,cv2.COLOR_BGR2GRAY)

    #kesilmis_kare_canny = cv2.Canny(kesilmis_kare_gri,30,150)
    #kesilmis_kare_canny = cv2.dilate(kesilmis_kare_canny,kernel,iterations=2)
    #kesilmis_kare_canny = cv2.morphologyEx(kesilmis_kare_canny, cv2.MORPH_CLOSE, kernel)

    kesilmis_kare_hsv = cv2.cvtColor(kesilmis_kare,cv2.COLOR_BGR2HSV)

    alt_deger = np.array([0,48,80]) #0 48 80
    ust_deger = np.array([20,255,255])  #20 255 255

    renk_filtresi_sonuc = cv2.inRange(kesilmis_kare_hsv,alt_deger,ust_deger)
    renk_filtresi_sonuc = cv2.morphologyEx(renk_filtresi_sonuc, cv2.MORPH_CLOSE,kernel)

    sonuc = kesilmis_kare.copy()

    _,cnts, hierarchy   = cv2.findContours(renk_filtresi_sonuc,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    max_genislik = 0
    max_uzunluk = 0
    max_index = -1
    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h = cv2.boundingRect(cnt)
        if(w>max_genislik and h>max_uzunluk): # en büyük kareyi almak için
            max_uzunluk = h
            max_genislik = w
            max_index = t

    if(len(cnts)>0): #kontür çizme işlemi
        x,y,w,h = cv2.boundingRect(cnts[max_index])
        cv2.rectangle(sonuc,(x,y),(x+w,y+h),[0,255,0],2)
        el_resim = renk_filtresi_sonuc[y:y+h,x:x+w]
        cv2.imshow("el resim", el_resim)
        print(Siniflandir(el_resim,veri_isimler,veri_resimler))

    cv2.imshow("kare",kare)
    cv2.imshow("kesilmiş",kesilmis_kare)
    cv2.imshow("renk filtresi ",renk_filtresi_sonuc)
    cv2.imshow("sonuc",sonuc)

    #cv2.imshow("kesilmis kare",kesilmis_kare_treshold)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


kamera.release()
cv2.destroyAllWindows()
