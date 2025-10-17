import random
from typing import List, Dict

from pycocotools.coco import COCO
import os.path
from pathlib import Path

from scripts import coco_process, crop_coco, dental_preprocessing, coco2yolo, split_coco_dataset


def generate_dataset_splits(
        all_image_ids: List[int],
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
) -> Dict[str, List[int]]:
    """
    接收一个包含所有图像ID的列表，并将其划分为训练池、验证集和测试集。
    这是一个通用的划分函数，与数据格式无关。

    :param all_image_ids: 包含数据集中所有图像ID的列表。
    :param val_ratio: 验证集比例。
    :param test_ratio: 测试集比例。
    :param seed: 随机种子，确保可复现性。
    :return: 一个字典，包含 'train_pool', 'val', 'test' 的ID列表。
    """
    random.seed(seed)
    random.shuffle(all_image_ids)

    test_size = int(len(all_image_ids) * test_ratio)
    val_size = int(len(all_image_ids) * val_ratio)

    test_ids = all_image_ids[:test_size]
    val_ids = all_image_ids[test_size: test_size + val_size]
    train_pool_ids = all_image_ids[test_size + val_size:]

    print("\n通用数据集ID划分完毕:")
    print(f"  - 训练池 (Train Pool): {len(train_pool_ids)} 张")
    print(f"  - 验证集 (Validation): {len(val_ids)} 张")
    print(f"  - 测试集 (Test): {len(test_ids)} 张")

    return {
        'train': train_pool_ids,
        'val': val_ids,
        'test': test_ids
    }


if __name__ == '__main__':
    coco_raw_path = 'raw_data/annotations.json'
    coco_merge_path = 'raw_data/merged_annotations.json'

    # # 1.对coco文件进行处理,将多生牙统一合并为同一个标注
    # coco_process.delete_category(
    #     coco_path=coco_raw_path,
    #     output_path=coco_merge_path,
    #     delete_names=['18','28','38','48'],
    #     dsy_names = ['9','91','92','93','94','95','96']
    # )

    img_path = 'raw_data'

    img_crop_path = 'data/dataset/coco/crop_child'
    # 2.对coco数据进行裁剪,重新生成
    # coco_crop_merge_path = crop_coco.crop_coco_dataset(coco_merge_path, img_path, img_crop_path)
    img_crop_preprocessing_path = 'data/dataset/coco/crop_child/preprocessing_images'
    # 3.对数据进行增强
    #dental_preprocessing.dental_preprocessing_pipeline(os.path.join(img_crop_path,'images'), img_crop_preprocessing_path)

    coco_for_ids = COCO('data/dataset/coco/crop_child/annotations/annotations.json')
    all_ids = coco_for_ids.getImgIds()

    # 对数据进行划分
    master_split = generate_dataset_splits(
        all_image_ids=all_ids,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )


    # # 4.将coco数据集转换为yolo
    yolo_path = 'data/dataset/yolo/process'
    yolo_path_raw = 'data/dataset/yolo/raw'
    new_ann_path = os.path.join(img_crop_path,'annotations','annotations.json')
    coco2yolo.coco_to_yolo_dataset(new_ann_path, img_crop_preprocessing_path,yolo_path,master_split)
    coco2yolo.coco_to_yolo_dataset(new_ann_path,'data/dataset/coco/crop_child/images',yolo_path_raw,master_split)

    # 5.对coco数据集进行划分
    coco_split_ann_path = 'data/dataset/coco/crop_child/annotations'
    split_coco_dataset.create_coco_efficiency_files( Path('data/dataset/coco/crop_child/annotations/annotations.json'),
        output_dir=Path(coco_split_ann_path),
        master_split=master_split,
        percentages=[20, 40, 60, 80, 100])

