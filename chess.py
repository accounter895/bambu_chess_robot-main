import cv2
import numpy as np

def dress_detect(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # 转换为灰度图像
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=50, param2=55)
    if circles is not None:
        circles = np.uint16(np.around(circles))  # 对圆的圆心坐标和半径四舍五入取整
    
    black_chess_coords = []
    white_chess_coords = []

    for i in circles[0, :]:
        # 提取圆形区域
        mask = np.zeros_like(img_gray)
        cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
        
        # 计算圆形区域的平均灰度值
        mean_val = cv2.mean(img_gray, mask=mask)[0]
        
        if mean_val < 128:  # 黑色棋子
            black_chess_coords.append((i[0], i[1]))
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 用红色标记黑色棋子
        else:  # 白色棋子
            white_chess_coords.append((i[0], i[1]))
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 用蓝色标记白色棋子

    print("Black Chess Coordinates:", black_chess_coords)
    print("White Chess Coordinates:", white_chess_coords)

import cv2
import numpy as np

def dress_borad_detect(img):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 对轮廓进行多边形逼近
        epsilon = 0.1 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:  # 确保是四边形
            # 获取四个角点的坐标
            corners = [tuple(point[0]) for point in approx]
            
            print("Chessboard Corners:", corners)
            
            # 在原图像上绘制角点
            for corner in corners:
                cv2.circle(img, corner, 5, (0, 255, 0), -1)
            
            # 计算棋盘的中心点
            center_x = int(sum([corner[0] for corner in corners]) / 4)
            center_y = int(sum([corner[1] for corner in corners]) / 4)
            print("Chessboard Center:", (center_x, center_y))
            
            # 定义目标正方形的四个角点
            target_corners = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
            src_corners = np.float32(corners)
            
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(src_corners, target_corners)
            
            # 应用透视变换
            warped_img = cv2.warpPerspective(img, M, (300, 300))
            
            # 计算正方形区域内每个小格子的中心点
            width = 100
            height = 100
            center_points_warped = []
            for i in range(3):
                for j in range(3):
                    x = int(j * width + width / 2)
                    y = int(i * height + height / 2)
                    center_points_warped.append((x, y))
                    cv2.circle(warped_img, (x, y), 3, (255, 255, 0), -1)
            
            # 将中心点逆变换回原始图像坐标系
            center_points = []
            for point in center_points_warped:
                x, y = point
                src_point = np.array([[x, y]], dtype=np.float32)
                dst_point = cv2.perspectiveTransform(src_point.reshape(-1, 1, 2), np.linalg.inv(M))[0][0]
                center_points.append((int(dst_point[0]), int(dst_point[1])))
                cv2.circle(img, (int(dst_point[0]), int(dst_point[1])), 3, (255, 255, 0), -1)
            
            print("Center Points of Chessboard Grids:", center_points)

if __name__ == '__main__':
    print("开始测试")
    img_load = 'img\chessboard_f2.jpg'
    img0 = cv2.imread(img_load)
    img_ori = cv2.resize(img0, None, fx=1/4, fy=1/4)
    cv2.imshow('chessboard_f2', img_ori)
    dress_detect(img_ori)
    dress_borad_detect(img_ori)
    cv2.imshow('Detected Circles and Chessboard Corners', img_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

