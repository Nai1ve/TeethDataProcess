import json
import random
from pathlib import Path
from typing import Dict, List, Any
from pycocotools.coco import COCO



def _create_json_files(
    full_coco_data: Dict[str, Any],
    split_definition: Dict[str, List[int]],
    output_dir: Path
) -> None:
    for subset_name, subset_ids in split_definition.items():
        print(f"\n  正在生成 '{subset_name}.json'...")
        subset_id_set = set(subset_ids)
        subset_images = [img for img in full_coco_data['images'] if img['id'] in subset_id_set]
        subset_annotations = [ann for ann in full_coco_data['annotations'] if ann['image_id'] in subset_id_set]
        subset_data = {"info": full_coco_data.get("info", {}), "licenses": full_coco_data.get("licenses", []), "categories": full_coco_data["categories"], "images": subset_images, "annotations": subset_annotations}
        output_path = output_dir / f"{subset_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subset_data, f, indent=4)
        print(f"  ✅ 已生成 '{subset_name}.json': {len(subset_images)} 张图片, {len(subset_annotations)} 个标注。")


def create_coco_efficiency_files(
        annotation_path: Path,
        output_dir: Path,
        master_split: Dict[str, List[int]],
        percentages: List[int] = [20, 40, 60, 80, 100]
) -> None:
    """
    接收一个预先划分好的ID字典，为COCO数据集生成效率实验文件。

    :param annotation_path: 原始COCO完整标注文件的路径。
    :param output_dir: 输出目录。
    :param master_split: 预先划分好的ID字典，必须包含 'train_pool', 'val', 'test' 键。
    :param percentages: 需要生成的训练集百分比列表。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在加载原始COCO标注文件: {annotation_path}")
    coco = COCO(str(annotation_path))
    data = coco.dataset
    print("加载完成。")

    # 从master_split中提取ID
    val_ids = master_split['val']
    test_ids = master_split['test']
    train_pool_ids = master_split['train']

    # 生成固定的验证集和测试集JSON
    print("\n正在生成固定的COCO验证集和测试集文件...")
    _create_json_files(data, {'val': val_ids, 'test': test_ids}, output_dir)

    # 根据百分比生成训练集JSON
    print("\n正在根据百分比生成COCO训练集文件...")
    random.shuffle(train_pool_ids)  # 再次打乱以保证切片的随机性

    for p in sorted(percentages):
        num_to_sample = int(len(train_pool_ids) * (p / 100))
        subset_train_ids = train_pool_ids[:num_to_sample]
        subset_name = f"train_{p}"
        _create_json_files(data, {subset_name: subset_train_ids}, output_dir)

    print("\n🎉 所有COCO效率实验数据集已生成完毕！")


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

