import json
import os
import cv2
import numpy as np
from torchvision.transforms.v2.functional import crop_image
from tqdm import tqdm
from pycocotools.coco import COCO

def crop_coco_dataset(anno_path,img_dir,output_dir,top_ratio = 0.15 ,left_ratio = 0.12,right_ratio = 0.12):
    """
    裁剪COCO数据集图像并调整标注框
    :param anno_path: COCO标注文件路径
    :param img_dir: 原始图像目录
    :param output_dir: 输出目录
    :param top_ratio: 上方裁剪比例
    :param left_ratio: 左侧裁剪比例
    :param right_ratio: 右侧裁剪比例
    """

    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    # 加载COCO标注
    coco = COCO(anno_path)
    with open(anno_path, 'r') as f:
        ann_data = json.load(f)

    new_ann_data = {
        "info": ann_data["info"],
        "licenses": ann_data["licenses"],
        "categories": ann_data["categories"],
        "images": [],
        "annotations": []
    }

    # 处理每张图像
    for img_id in tqdm(coco.getImgIds(), desc='Processing images'):
        img_info = coco.loadImgs(img_id)[0]
        img_file_name = img_info['file_name']

        img_path = os.path.join(img_dir, img_file_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Image not found: {img_path}")
            continue


        h,w = img.shape[:2]
        top = int(h * top_ratio)
        left = int(w * left_ratio)
        right = w - int(w * right_ratio)
        bottom = h


        # 裁剪
        crop_img = img[top:bottom,left:right] # 切片

        # 保存裁剪后的图片
        new_img_path = os.path.join(output_dir,'images', img_file_name)
        cv2.imwrite(new_img_path,crop_img)

        # 更新图像尺寸信息
        new_img_info = img_info.copy()
        new_img_info['height'] = crop_img.shape[0]
        new_img_info['width'] = crop_img.shape[1]
        new_ann_data['images'].append(new_img_info)


        # 处理标注框信息
        ann_ids = coco.getAnnIds(imgIds= img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            # 计算新坐标
            x,y,bw,bh = ann['bbox']
            new_x =max(0,int(x - left))
            new_y = max(0,int(y - top))
            # 防止裁剪区域覆盖到边界框
            new_w = min(bw,right - left - new_x)
            new_h = min(bh,bottom - top - new_y)
            new_area = new_w * new_h
            ann['area'] = new_area

            # 检查框是否有效
            if new_w > 0 and new_h >0:
                new_ann = ann.copy()
                new_ann['bbox'] = [new_x,new_y,new_w,new_h]

                # 更新分割标注
                if 'segmentation' in new_ann:
                    for seg in new_ann['segmentation']:
                        for i in range(0, len(seg), 2):
                            seg[i] = max(0, seg[i] - left)
                            seg[i+1] = max(0, seg[i+1] - top)

                new_ann_data['annotations'].append(new_ann)

    # 保存标注文件
    new_anno_path = os.path.join(output_dir,'annotations','annotations.json')
    with open(new_anno_path ,'w') as f:
        json.dump(new_ann_data,f)

    print(f"Processing complete! Cropped images saved to: {os.path.join(output_dir, 'images')}")
    print(f"Updated annotations saved to: {new_anno_path}")
    return new_anno_path
