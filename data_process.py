import os.path

from scripts import coco_process, crop_coco, dental_preprocessing, coco2yolo, split_coco_dataset

if __name__ == '__main__':
    coco_raw_path = 'data/child/annotations.json'
    coco_merge_path = 'data/child/merged_annotations.json'

    # 1.对coco文件进行处理,将多生牙统一合并为同一个标注
    coco_process.merge_to_existing_category(
        coco_path=coco_raw_path,
        output_path=coco_merge_path,
        merge_names=["91", "92","93"],
        target_name="9"
    )

    img_path = 'data/child'

    img_crop_path = 'data/dataset/coco/crop_child'

    # 2.对coco数据进行裁剪,重新生成
    crop_coco.crop_coco_dataset(coco_merge_path, img_path, img_crop_path)
    img_crop_preprocessing_path = 'data/dataset/coco/crop_child/preprocessing_images'
    # 3.对数据进行增强
    dental_preprocessing.dental_preprocessing_pipeline(os.path.join(img_crop_path,'images'), img_crop_preprocessing_path)
    # 4.将coco数据集转换为yolo
    yolo_path = 'data/dataset/yolo/process'
    yolo_path_raw = 'data/dataset/yolo/raw'
    new_ann_path = os.path.join(img_crop_path,'annotations','annotations.json')
    coco2yolo.coco_to_yolo_dataset(new_ann_path, img_crop_preprocessing_path, yolo_path)
    coco2yolo.coco_to_yolo_dataset(new_ann_path,'data/dataset/coco/crop_child/images',yolo_path_raw)

    # 5.对coco数据集进行划分
    coco_split_ann_path = 'data/dataset/coco/crop_child/annotations'
    split_coco_dataset.split_coco_dataset(coco_merge_path, coco_split_ann_path)

