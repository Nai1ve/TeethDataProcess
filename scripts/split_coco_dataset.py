import json
import random
from pycocotools.coco import COCO


def split_coco_dataset(annotation_path, output_dir, ratios=(0.8, 0.1, 0.1), seed=42):
    """
    划分 COCO 数据集，生成训练集、验证集和测试集的 JSON 文件
    :param annotation_path: 原始 COCO 标注文件路径（如 instances_train2017.json）
    :param output_dir: 输出目录
    :param ratios: 比例元组 (train_ratio, val_ratio, test_ratio)，总和需为 1
    :param seed: 随机种子，确保可复现
    """
    # 设置随机种子
    random.seed(seed)

    # 加载原始标注
    coco = COCO(annotation_path)
    data = coco.dataset

    # 获取所有图像 ID 并打乱顺序
    img_ids = [img['id'] for img in data['images']]
    random.shuffle(img_ids)

    # 计算各子集数量
    total = len(img_ids)
    num_train = int(total * ratios[0])
    num_val = int(total * ratios[1])
    num_test = total - num_train - num_val

    # 划分图像 ID
    train_ids = img_ids[:num_train]
    val_ids = img_ids[num_train:num_train + num_val]
    test_ids = img_ids[num_train + num_val:]

    # 构建子集标注数据
    subsets = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    for subset_name, subset_ids in subsets.items():
        # 筛选图像
        subset_images = [img for img in data['images'] if img['id'] in subset_ids]

        # 筛选标注（仅保留子集图像中的标注）
        subset_annotations = [
            ann for ann in data['annotations']
            if ann['image_id'] in subset_ids
        ]

        # 构建子集 JSON 结构
        subset_data = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "categories": data["categories"],  # 保留所有类别
            "images": subset_images,
            "annotations": subset_annotations
        }

        # 保存子集 JSON
        output_path = f"{output_dir}/{subset_name}.json"
        with open(output_path, 'w') as f:
            json.dump(subset_data, f)
        print(f"已生成 {subset_name} 集: {len(subset_ids)} 张图片, {len(subset_annotations)} 个标注")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="COCO 数据集划分工具")
    # parser.add_argument('--json', type=str, required=True, help='原始 COCO 标注文件路径')
    # parser.add_argument('--output', type=str, required=True, help='输出目录路径')
    # parser.add_argument('--ratios', type=float, nargs=3, default=[0.7, 0.2, 0.1],
    #                     help='划分比例（训练集 验证集 测试集），例如 0.7 0.2 0.1')
    # parser.add_argument('--seed', type=int, default=42, help='随机种子（默认：42）')
    # args = parser.parse_args()

    annotation_path = '../data/dataset/coco/crop_child/annotations/merged_annotations.json'
    output_dir = '../data/dataset/coco/crop_child/annotations'

    split_coco_dataset(
        annotation_path=annotation_path,
        output_dir=output_dir,
    )