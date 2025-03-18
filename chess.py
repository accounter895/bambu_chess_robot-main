import cv2
import numpy as np

# 读取图像并缩小
img0 = cv2.imread('img\chessboard_f1.jpg')
img = cv2.resize(img0, None, fx=1/2, fy=1/2)
cv2.imshow('circle2', img)

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测圆形
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=50, param2=50)

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    black_chess_coords = []
    white_chess_coords = []

    for i in circles[0, :]:
        # 提取圆形区域
        mask = np.zeros_like(gray)
        cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
        
        # 计算圆形区域的平均灰度值
        mean_val = cv2.mean(gray, mask=mask)[0]
        
        if mean_val < 128:  # 黑色棋子
            black_chess_coords.append((i[0], i[1]))
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 用红色标记黑色棋子
        else:  # 白色棋子
            white_chess_coords.append((i[0], i[1]))
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 用蓝色标记白色棋子

    print("Black Chess Coordinates:", black_chess_coords)
    print("White Chess Coordinates:", white_chess_coords)

    cv2.imshow('Detected Circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No circles detected.")