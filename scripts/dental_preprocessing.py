import numpy as np
import os
import cv2

def denoise_image(img):
    """非局部均值去燥 + 中值滤波组合"""
    denoised = cv2.fastNlMeansDenoising(
        img,None,
        h = 7,                # 亮度分量强度（7-15适用于牙片）
        templateWindowSize=7, # 局部区域大小
        searchWindowSize=21   # 搜索范围（默认值平衡效果与速度）
    )
    # 中值滤波去除椒盐噪声
    denoised = cv2.medianBlur(denoised,3)
    return denoised

def enhance_contrast(img):
    """CLAHE自适应直方图均衡化"""

    # Lab空间处理亮度通道（避免颜色失真）
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    # 对亮度通道做CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0, # 对比度限制（牙根区域用2.0-3.0）
        tileGridSize=(8,8) # 局部区域大小（小区域增强细节）
    )
    l_enhanced = clahe.apply(l)

    #合并通道并转会BGR
    lab_enhanced = cv2.merge((l_enhanced,a,b))
    return cv2.cvtColor(lab_enhanced,cv2.COLOR_LAB2BGR)


def dental_preprocessing_pipeline(dir_path,output_dir):
   os.makedirs(output_dir,exist_ok=True)
   for file in os.listdir(dir_path):
    if file.endswith('.jpg'):
        img_path = os.path.join(dir_path,file)
        print(f'开始处理：{file}')
        # 读取图像
        img = cv2.imread(img_path)
        if len(img.shape) == 2:  # 灰度图转BGR三通道
            print(f"path:{img_path} covert 2 BGR")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 去噪
        img1 = denoise_image(img)

        # 对比度增强
        img2 = enhance_contrast(img1)

        # 边缘锐华
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 9, -1],
             [-1, -1, -1]]
        )

        img3 = cv2.filter2D(img2, -1, kernel)

        filename = img_path.split('/')[-1]

        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, img3)
        # imgs = [img,img1,img2,img3]
        #
        # combined = np.hstack(imgs)
        #
        # cv2.imshow("Comparison", combined)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(f'完成：{file}的处理')