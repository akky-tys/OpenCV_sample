import cv2
import sys

args = sys.argv
targetImage = args[1]

# 分類器の指定するよ
##cascade_file = "haarcascade_frontalface_alt2.xml"
face_cascade_file = "..//openENV/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
smile_cascade_file = "..//openENV/lib/python3.7/site-packages/cv2/data/haarcascade_smile.xml"
##cascade_file = "..//openENV/lib/python3.7/site-packages/cv2/data/haarcascade_fullbody.xml"
eyes_cascade_file = "..//openENV/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"
face_cascade      = cv2.CascadeClassifier(face_cascade_file)
smile_cascade      = cv2.CascadeClassifier(smile_cascade_file)
eyes_cascade      = cv2.CascadeClassifier(eyes_cascade_file)

#判別したい画像を読み込み
img = cv2.imread(targetImage)


##グレースケールに変換
##画像を判別する際に情報量が多すぎるとメモリを圧迫するから、
##カラーからグレーに変えて、情報を排除することで計算量を抑えている。

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#検知
faces = face_cascade.detectMultiScale(gray)
for (x,y,w,h) in faces:
  # 検知した物を囲む
  center = (x + w//2, y + h//2)
  cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
  # グレー
  faceROI_gray = gray[y:y+h, x:x+w]
  # カラー
  faceROI = img[y:y+h, x:x+w]

  
##  eyes = eyes_cascade.detectMultiScale(faceROI_gray)
##  for (x2,y2,w2,h2) in eyes:
##    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
##    radius = int(round((w2 + h2)*0.25))
##    cv2.circle(img, eye_center, radius, (255, 0, 0 ), 4)

  smiles = smile_cascade.detectMultiScale(faceROI_gray)
  if len(smiles) > 0 :
    for(smile_x, smile_y, smile_w, smile_h) in smiles:

      cv2.circle(img, (int(x+smile_x+smile_w/2),int(y+smile_y+smile_h/2)),int(smile_w/2), (0, 255, 0 ), 4)

cv2.imshow('img',img)
 
key = cv2.waitKey(0)
cv2.imwrite('mark_new.jpeg', img);
cv2.destroyAllWindows()
