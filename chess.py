import numpy as np
import cv2

def load_and_resize_image(image_path, size=(640, 320)):
    """加载并调整图像大小"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    return cv2.resize(image, size)

def detect_black_pieces(image):
    """检测黑棋子"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([179, 50, 80])
    
    mask = cv2.inRange(hsv_image, lower_black, upper_black)
    cv2.namedWindow('Black Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Black Mask', mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours.")
    
    black_piece_positions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        print(f"Contour area: {area}")
        if area > 50:  # 过滤掉小的噪声
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            black_piece_positions.append(center)
            cv2.circle(image, center, radius, (0, 255, 0), 2)
            print(f"Detected black piece at {center} with radius {radius}.")
    
    return black_piece_positions

def detect_white_pieces(image):
    """检测白棋子"""
    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义白色颜色的HSV范围
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 30, 255])
    
    # 创建掩码
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    white_piece_positions = []
    for contour in contours:
        # 计算轮廓的圆度
        area = cv2.contourArea(contour)
        if area > 50:  # 过滤掉小的噪声
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            white_piece_positions.append(center)
            cv2.circle(image, center, radius, (0, 0, 255), 2)
    
    return white_piece_positions

if __name__ == "__main__":
    # 加载并调整图像大小
    chess_ori = load_and_resize_image('img/chessboard_f1.jpg')
    if chess_ori is None:
        exit(1)
    
    # 检测黑棋子和白棋子
    black_pieces = detect_black_pieces(chess_ori)
    white_pieces = detect_white_pieces(chess_ori)
    
    print("Black pieces positions:", black_pieces)
    print("White pieces positions:", white_pieces)
    
    # 显示图像
    cv2.namedWindow('Color Scale Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Color Scale Image', chess_ori)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()