'''
功能：识别图片中的人脸并用绿色方框标注
模型：haar
优点：可以同时识别多个人脸
缺点：只能识别人的正脸，稍有角度都会影响识别
'''

# 1.导入库
import cv2
# 2.加载图片
img = cv2.imread('C:/Users/Bill/PycharmProjects/FaceRecognition/picture/image5.jpg')
# 3.加载人脸模型
face = cv2.CascadeClassifier("C:/Users/Bill/PycharmProjects/FaceRecognition/haarcascade_frontalface_alt.xml")
# 4.调整图片灰度(没必要识别颜色，灰度可以提高性能)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 5.检查人脸
faces = face.detectMultiScale(img_gray)
# 6.标记人脸
for(x, y, w, h) in faces:
    # 四个参数: 1.图片 2.坐标原点 3.识别大小 4.颜色RGB 5.线宽
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
# 7.创建窗口
cv2.namedWindow('Picture')
# 8.显示图片
cv2.imshow('HumanPicture', img)
# 9.暂停窗口
cv2.waitKey(0)
# 10.关闭窗口
cv2.destroyAllWindows()
