import cv2


def effect_face_func(img, rect):
    image_file  = "images/effect1.png"
    ##image_file  = "images/pengin.png"
    marge_image = cv2.imread(image_file)

    marge_image_gray = cv2.cvtColor(marge_image, cv2.COLOR_BGR2GRAY)

    (x1, y1, x2, y2) = rect

    w = x2 - x1
    h = y2 - y1
    
    img_face = cv2.resize(marge_image_gray, (w, h))
    img2 = img.copy()
    img2[y1:y2, x1:x2] = img_face
    return img2
 



# 分類器の指定
face_cascade_file = "..//openENV/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
smile_cascade_file = "..//openENV/lib/python3.7/site-packages/cv2/data/haarcascade_smile.xml"
eyes_cascade_file = "..//openENV/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"
face_cascade      = cv2.CascadeClassifier(face_cascade_file)
smile_cascade      = cv2.CascadeClassifier(smile_cascade_file)
eyes_cascade      = cv2.CascadeClassifier(eyes_cascade_file)


# カメラ映像取得
cap = cv2.VideoCapture(0)

# 初期フレームの読込
ret, frame = cap.read()
height, width, channels = frame.shape

# 変換処理ループ
while ret == True:
  img = frame
  # 画像の取得と顔の検出
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face_list = face_cascade.detectMultiScale(gray)

  # 検出した顔に印を付ける
  for (x, y, w, h) in face_list:
    center = (x + w//2, y + h//2)
##    cv2.ellipse(gray, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    # グレー
    faceROI_gray = gray[y:y+h, x:x+w]
    gray = effect_face_func(gray, (x+5, y+5, x+w+5, y+h+5))

##  eyes = eyes_cascade.detectMultiScale(faceROI_gray)
##  for (x2,y2,w2,h2) in eyes:
##    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
##    radius = int(round((w2 + h2)*0.25))
##    cv2.circle(img, eye_center, radius, (255, 0, 0 ), 4)

##    smiles = smile_cascade.detectMultiScale(faceROI_gray)
##    if len(smiles) > 0 :
##      for(smile_x, smile_y, smile_w, smile_h) in smiles:
##        cv2.circle(gray, (int(x+smile_x+smile_w/2),int(y+smile_y+smile_h/2)),int(smile_w/2), (0, 255, 0 ), 4)



  # フレーム表示
  cv2.imshow('img', gray)

  # Escキーで終了
  key = cv2.waitKey(30)
  if key == 27:
      break

  # 次のフレーム読み込み
  ret, frame = cap.read()


cv2.destroyAllWindows()
cap.release()


