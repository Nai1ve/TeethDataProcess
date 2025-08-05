import json
import os
import random
import shutil
import argparse
from tqdm import tqdm
import yaml
from PIL import Image, ImageDraw
import numpy as np


def coco_to_yolo_dataset(
        coco_json_path,
        image_dir,
        output_dir,
        ratios=(0.8, 0.1, 0.1),
        seed=34,
):
    """
    将未划分的 COCO 数据集转换为 YOLO 格式并进行划分

    参数:
        coco_json_path: COCO JSON 文件路径
        image_dir: 图像目录路径
        output_dir: 输出目录路径
        ratios: (训练集比例, 验证集比例, 测试集比例)
        seed: 随机种子
        create_visualization: 是否创建可视化预览
    """
    # 验证比例总和为1
    if abs(sum(ratios) - 1.0) > 0.01:
        raise ValueError(f"比例总和必须为1.0，当前为: {sum(ratios)}")

    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    for folder in ['images', 'labels']:
        for subset in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, folder, subset), exist_ok=True)

    # 加载 COCO 数据
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 1. 提取类别信息
    categories = coco_data['categories']
    nc = len(categories)

    # 创建类别映射：category_id -> 类别名称
    id_to_name = {cat['id']: cat['name'] for cat in categories}

    # 按 category_id 排序获取类别列表
    sorted_categories = sorted(categories, key=lambda x: x['id'])
    class_names = [cat['name'] for cat in sorted_categories]
    class_ids = [cat['id'] for cat in sorted_categories]

    # 创建类别映射：category_id -> YOLO 类别索引 (0-based)
    cat_id_to_yolo_id = {cat['id']: idx for idx, cat in enumerate(sorted_categories)}

    print(f"📊 数据集统计:")
    print(f"  类别数量: {nc}")
    print(f"  类别列表: {', '.join(class_names)}")

    # 2. 提取图像信息
    images = coco_data['images']

    # 按 image_id 映射图像信息
    image_id_to_info = {img['id']: img for img in images}

    # 3. 按 image_id 分组标注
    image_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        image_id_to_anns[img_id].append(ann)

    # 4. 随机划分图像
    all_image_ids = list(image_id_to_info.keys())
    random.seed(seed)
    random.shuffle(all_image_ids)

    total_images = len(all_image_ids)
    train_end = int(ratios[0] * total_images)
    val_end = train_end + int(ratios[1] * total_images)

    splits = {
        'train': all_image_ids[:train_end],
        'val': all_image_ids[train_end:val_end],
        'test': all_image_ids[val_end:]
    }



    # 6. 处理每个图像
    print("\n🔧 转换标注并复制文件...")
    class_counts = {subset: {cls_id: 0 for cls_id in class_ids} for subset in splits}

    for subset, image_ids in splits.items():
        print(f"  处理 {subset} 集 ({len(image_ids)} 个图像)")

        for img_id in tqdm(image_ids, desc=f"{subset} 集"):
            img_info = image_id_to_info[img_id]
            file_name = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            # 源图像路径
            img_src = os.path.join(image_dir, file_name)

            # 目标图像路径
            img_dest = os.path.join(output_dir, 'images', subset, file_name)

            # 复制图像文件
            try:
                shutil.copy2(img_src, img_dest)

                # 标签文件路径
                base_name = file_name.split('.')[0]
                label_path = os.path.join(output_dir, 'labels', subset, f"{base_name}.txt")

                # 获取该图像的标注
                anns = image_id_to_anns.get(img_id, [])

                # 创建标签文件
                with open(label_path, 'w') as f:
                    for ann in anns:
                        # 获取类别ID并转换为YOLO索引
                        cat_id = ann['category_id']
                        yolo_cls_id = cat_id_to_yolo_id[cat_id]

                        # 转换边界框 [x, y, width, height] -> [x_center, y_center, width, height]
                        bbox = ann['bbox']
                        x, y, w, h = bbox

                        # 边界检查
                        x = max(0, min(x, img_width - 1))
                        y = max(0, min(y, img_height - 1))
                        w = min(w, img_width - x)
                        h = min(h, img_height - y)

                        # 归一化
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width = w / img_width
                        height = h / img_height

                        # 写入YOLO格式
                        f.write(f"{yolo_cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                        # 更新类别统计
                        class_counts[subset][cat_id] += 1

                # 如果没有标注，创建空文件
                if not anns:
                    open(label_path, 'a').close()
            except Exception as e:
                print(f'操作文件失败:{e}')

    # 7. 创建 data.yaml 配置文件
    yaml_content = {
        'path': os.path.relpath(output_dir),
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'test': os.path.join('images', 'test'),

        'nc': nc,
        'names': class_names,

        'class_ids': class_ids,

        'dataset_stats': {
            'total_images': total_images,
            'subsets': {}
        }
    }

    # 添加子集统计信息
    for subset in splits:
        yaml_content['dataset_stats']['subsets'][subset] = {
            'image_count': len(splits[subset]),
            'class_distribution': {
                id_to_name[cat_id]: class_counts[subset][cat_id]
                for cat_id in class_ids
            }
        }

    # 保存YAML文件
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ COCO转YOLO数据集完成!")
    print(f"📁 输出目录: {os.path.abspath(output_dir)}")
    print(f"📄 配置文件: {yaml_path}")

    # 打印详细统计
    print("\n📊 数据集划分统计:")
    for subset in splits:
        count = len(splits[subset])
        print(f"  {subset.upper()}: {count} 个图像 ({count / total_images:.1%})")
        for cat_id in class_ids:
            cat_name = id_to_name[cat_id]
            c_count = class_counts[subset][cat_id]
            if c_count > 0:
                print(f"    {cat_name}: {c_count} 个标注")

    return yaml_path
