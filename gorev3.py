
import cv2
import numpy as np

image = cv2.imread("renklifoto.jpg")
if image is None:
    print("Hata: 'renklifoto.jpg' bulunamadi!")
    exit()

opencv_gri = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def manuel_bgr_to_gray(bgr_img):
    b = bgr_img[:, :, 0].astype(np.float32)
    g = bgr_img[:, :, 1].astype(np.float32)
    r = bgr_img[:, :, 2].astype(np.float32)
    gri = 0.114 * b + 0.587 * g + 0.299 * r
    return np.round(gri).astype(np.uint8)

manuel_gri = manuel_bgr_to_gray(image)

mutlak_fark = cv2.absdiff(opencv_gri, manuel_gri)
toplam_hata = np.sum(mutlak_fark)
ortalama_piksel_hatasi = np.mean(mutlak_fark)

print(f"Toplam Mutlak Hata (SAD): {toplam_hata}")
print(f"Piksel Basina Ortalama Hata: {ortalama_piksel_hatasi:.5f}")

cv2.imwrite("gri_opencv.jpg", opencv_gri)
cv2.imwrite("gri_manuel.jpg", manuel_gri)
cv2.imwrite("gri_fark.jpg", mutlak_fark)

cv2.imshow("1. OpenCV Gri", opencv_gri)
cv2.imshow("2. Manuel Gri", manuel_gri)
cv2.imshow("3. Mutlak Fark", mutlak_fark)

cv2.waitKey(0)
cv2.destroyAllWindows()

