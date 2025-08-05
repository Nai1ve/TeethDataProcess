import cv2
from pycocotools.coco import COCO
import os

dataset_dir = '../data/dataset/coco/crop_child/preprocessing_images'

dist_dir = '../data/analysis/'

def seg_teeth_from_data():
    """
        从数据集中分割出乳牙图片
    """

    ann_file_path = r'../data/dataset/coco/crop_child/annotations/annotations.json'
    seg_teeth(dataset_dir,ann_file_path,dist_dir)
    # for file in os.listdir(dataset_dir):
    #     file_path = os.path.join(dataset_dir,file)
    #     if os.path.isdir(file_path):
    #         ann_file_path = os.path.join(file_path,f'{file}.json')
    #         seg_teeth(file_path,ann_file_path,dist_dir)

def seg_teeth(source_dir,ann_file_path,to_path):
    """
    将数据按照标注拆分后写入指定文件夹
    """
    coco = COCO(ann_file_path)


    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(img_id)
        annotations = coco.loadAnns(ann_ids)
        file_name = img_info['file_name']
        print(f'processing {file_name}')

        img_path = os.path.join(source_dir,file_name)
        img = cv2.imread(img_path)

        for ann in annotations:
            # 获取坐标
            x,y,w,h = ann['bbox']
            # print(x,y,w,h)
            x,y,w,h = int(x),int(y),int(w),int(h)

            # 获取标注图像
            cropped_image = img[y:y+h,x:x+w]
            # cv2.imshow("1",cropped_image)
            # cv2.waitKey(0)

            category_id = ann['category_id']
            category_name = coco.loadCats(category_id)[0]['name']

            dir_path = os.path.join(to_path,category_name)
            output_path = os.path.join(dir_path,f"{file_name[:-4]}_{category_name}_{ann['id']}.jpg")

            if not os.path.exists(dir_path):
                print(f"create file :{dir_path}")
                os.makedirs(dir_path,exist_ok=True)

            cv2.imwrite(output_path,cropped_image)
            print(f"Saved {output_path}")




if __name__ == '__main__':
    seg_teeth_from_data()
