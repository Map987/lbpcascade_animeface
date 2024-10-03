import cv2
import sys

def detect_faces(image_path):
    # 加载人脸识别的预训练模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_animeface.xml')

    # 读取图片
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 如果检测到人脸，返回True
    if len(faces) > 0:
        print('face detected')
        return True
    else:
        print('no face detected')
        return False

if __name__ == '__main__':
    image_path = sys.argv[1]
    detect_faces(image_path)
