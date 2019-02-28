'''
功能：调用摄像头，识别人脸，绿色方框标注
模型：haar
缺点：只能识别竖直方向的人脸（应该跟像素刷新方向有关）
优点：可以同时识别多个人脸

'''
# 1.导入库
import cv2
# 2.加载人脸模型
face = cv2.CascadeClassifier("C:/Users/Bill/PycharmProjects/FaceRecognition/haarcascade_frontalface_alt.xml")
# 3.打开摄像头
capture = cv2.VideoCapture(0)
# 4.创建窗口
cv2.namedWindow('Picture')
# 5.获取摄像头实时画面
while True:
    # 5.1读取摄像头的帧画面
    ret, frame = capture.read()
    # 5.2图片灰度调整
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 5.3检查人脸
    faces = face.detectMultiScale(img_gray)
    # 5.4标记人脸
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 5.5 显示图片
        cv2.imshow('CoolMan', frame)
    # 5.6 暂停窗口
        if cv2.waitKey(5) & 0xff == ord('q'):
            break
# 6.释放资源
capture.release()
# 7.关闭窗口
cv2.destroyAllWindows()
