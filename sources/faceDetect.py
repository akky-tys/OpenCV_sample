import cv2
import numpy as np

def effect_image_func(img, rect):
    image_file  = "images/effect2.png"
    marge_image = cv2.imread(image_file)

    marge_image_gray = cv2.cvtColor(marge_image, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    (x1, y1, x2, y2) = rect
    print(x1)
    print(y1)

    defW = 100;
    defH = 100;

    diffW = x1 - defW;
    diffH = y2 - defH;

        
    if diffW < 0 :
      defW  = defW - x1;

    if diffH < 0 :
      defH  = defH - x1;
    
       
    img_face = cv2.resize(marge_image_gray, (defW, defH))
    ## (test1, test2) = img_face.shape;
    print(img_face.shape);
    img2 = img.copy()
    
    img2[x1 - defW:x1, y1-defH:y1] = img_face
    return img2
 

def effect_face_func(img, rect):
    image_file  = "images/effect2.png"
    marge_image = cv2.imread(image_file)

    marge_gray = cv2.cvtColor(marge_image, cv2.COLOR_BGR2GRAY)

    (x1, y1, x2, y2) = rect
    
    w = x2 - x1
    h = y2 - y1
    
    img_face = cv2.resize(marge_gray, (w, h))
    
    img2 = img.copy()
    img2[y1:y2, x1:x2]      = img_face
    return img2
 


# 定数定義
ESC_KEY     = 27     # Escキー
INTERVAL    = 33     # 待ち時間
FRAME_RATE  = 30  # fps
WINDOW_NAME = "faceWindow"
DEVICE_ID   = 0     # カメラのデバイスIDだよ 

# 分類器の指定するよ
##cascade_file = "haarcascade_frontalface_default.xml"
cascade_file = "..//openENV/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_file)


# カメラから映像を取得するよ
cap = cv2.VideoCapture(DEVICE_ID)

# 初期フレームの読込
end_flag, c_frame = cap.read()
height, width, channels = c_frame.shape

# 画像を表示するウィンドウを準備
cv2.namedWindow(WINDOW_NAME)


# 変換処理ループ
while end_flag == True:

    # 画像の取得と顔の検出
    img      = c_frame
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces    = face_cascade.detectMultiScale(gray, minSize=(100, 100))

    # 検出した顔の座標を使うよ
    for (x, y, w, h) in faces:
        ##顔に印をつける##
        color = (0, 0, 225)
        pen_w = 3

        # 印をつける場合
        ##cv2.rectangle(gray, (x, y), (x+w, y+h), color, thickness = pen_w)

        # effect をつける場合
        gray = effect_face_func(gray, (x, y, x+w, y+h))

        ##img_gray = effect_image_func(img_gray, (x+5, y+5, x+w+5, y+h+5))
    
    # フレームに表示をする
    cv2.imshow(WINDOW_NAME, gray)

    # Escキーで終了
    key = cv2.waitKey(INTERVAL)
    if key == ESC_KEY:
        break

    # 次のフレーム読み込み
    end_flag, c_frame = cap.read()

# 終了処理
cv2.destroyAllWindows()
cap.release()


