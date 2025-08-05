import json
import os
import yaml
from tqdm import tqdm
import shutil
import random

def create_yolo_dataset(
        coco_json_path,
        yolo_labels_dir,
        image_dir,
        output_dir,
        ratios=(0.8, 0.1, 0.1),
        seed=42
):
    """
    创建 YOLO 格式的数据集划分

    参数:
        coco_json_path: COCO 格式的 JSON 文件路径
        yolo_labels_dir: YOLO 格式的标签目录
        image_dir: 图像文件目录
        output_dir: 输出目录
        ratios: (训练集比例, 验证集比例, 测试集比例)
        seed: 随机种子
    """
    # 验证比例总和为1
    if abs(sum(ratios) - 1.0) > 0.01:
        raise ValueError(f"比例总和必须为1.0，当前为: {sum(ratios)}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    for folder in ['labels', 'images']:
        for subset in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, folder, subset), exist_ok=True)

    # 加载 COCO 数据
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 1. 提取类别信息
    categories = coco_data['categories']
    nc = len(categories)
    sorted_categories = sorted(categories, key=lambda x: x['id'])
    class_names = [cat['name'] for cat in sorted_categories]
    class_ids = [cat['id'] for cat in sorted_categories]

    print(f"📊 数据集统计:")
    print(f"  类别数量: {nc}")
    print(f"  类别列表: {', '.join(class_names)}")

    # 2. 构建图像列表
    images = coco_data['images']
    image_files = {}

    print("\n🔍 构建图像列表...")
    for img in tqdm(images):
        file_name = img['file_name']
        base_name = file_name.split('.')[0]

        # 查找对应的YOLO标签文件
        label_file = os.path.join(yolo_labels_dir, f"{base_name}.txt")

        # 检查图片文件是否存在
        img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(img_path):
            print(f"⚠️ 警告: 图片文件缺失 - {img_path}")
            continue

        # 检查标签文件是否存在
        if not os.path.exists(label_file):
            print(f"⚠️ 警告: 标签文件缺失 - {label_file}")
            continue

        image_files[base_name] = {
            'coco_image': img,
            'image_path': img_path,
            'label_path': label_file
        }

    total_images = len(image_files)
    print(f"✅ 找到 {total_images} 个有效图像-标签对")

    if total_images == 0:
        print("❌ 错误: 没有找到有效的图像-标签对")
        exit(1)

    # 3. 随机划分数据集
    random.seed(seed)
    all_keys = list(image_files.keys())
    random.shuffle(all_keys)

    train_end = int(ratios[0] * total_images)
    val_end = train_end + int(ratios[1] * total_images)

    splits = {
        'train': all_keys[:train_end],
        'val': all_keys[train_end:val_end],
        'test': all_keys[val_end:]
    }

    # 4. 复制文件并创建索引
    print("\n📂 复制文件到目标目录...")
    index_files = {}
    class_counts = {subset: {cls_id: 0 for cls_id in class_ids} for subset in splits}

    print(class_counts)

    for subset, keys in splits.items():
        index_files[subset] = []

        print(f"  处理 {subset} 集 ({len(keys)} 个样本)")
        for key in tqdm(keys, desc=f"{subset} 集"):
            data = image_files[key]

            # 复制图像文件
            img_file = data['coco_image']['file_name']
            img_src = data['image_path']
            img_dest = os.path.join(output_dir, 'images', subset, img_file)
            shutil.copy2(img_src, img_dest)

            # 复制标签文件
            label_src = data['label_path']
            label_file = os.path.basename(label_src)
            label_dest = os.path.join(output_dir, 'labels', subset, label_file)
            shutil.copy2(label_src, label_dest)

            # 添加到索引文件
            index_files[subset].append(os.path.abspath(img_dest))

            # 统计类别分布
            with open(label_src, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        cls_id = int(parts[0])
                        print(f'label:{label_src},cls_id:{cls_id}')
                        class_counts[subset][cls_id] += 1

        # 写入索引文件
        with open(os.path.join(output_dir, f"{subset}.txt"), 'w') as f:
            for path in index_files[subset]:
                f.write(path + "\n")

    # 5. 创建 data.yaml 配置文件
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': os.path.abspath(os.path.join(output_dir, 'train.txt')),
        'val': os.path.abspath(os.path.join(output_dir, 'val.txt')),
        'test': os.path.abspath(os.path.join(output_dir, 'test.txt')),

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
                class_names[i]: class_counts[subset][class_ids[i]]
                for i in range(nc)
            }
        }

    # 保存YAML文件
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\n✅ YOLO数据集创建完成!")
    print(f"📁 输出目录: {os.path.abspath(output_dir)}")
    print(f"📄 配置文件: {yaml_path}")

    # 打印详细统计
    print("\n📊 数据集划分统计:")
    for subset in splits:
        count = len(splits[subset])
        print(f"  {subset.upper()}: {count} 个图像 ({count / total_images:.1%})")
        for cls_name in class_names:
            cls_id = class_ids[class_names.index(cls_name)]
            c_count = class_counts[subset][cls_id]
            if c_count > 0:
                print(f"    {cls_name}: {c_count} 个标注")

    return yaml_path

create_yolo_dataset(
    coco_json_path='../data/child/merged_annotations.json',
    yolo_labels_dir='../data/dataset/yolo',
    image_dir='../data/dataset/coco/crop_child/images/images',
    output_dir='../data/dataset/yolo_child'
)