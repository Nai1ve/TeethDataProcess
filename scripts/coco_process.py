import json


def merge_to_existing_category(coco_path, output_path, merge_names, target_name):
    """
    :param merge_names: 待合并的类别名称列表（如 ["cat", "dog"]）
    :param target_name: 目标类别的名称（如 "animal"）
    """
    with open(coco_path, 'r') as f:
        data = json.load(f)

    # 1. 获取目标类别的 ID
    target_id = None
    for cat in data['categories']:
        if cat['name'] == target_name:
            target_id = cat['id']
            break
    if target_id is None:
        raise ValueError(f"目标类别 '{target_name}' 不存在！")

    # 2. 构建待合并类别的 ID 列表
    old_ids = []
    for cat in data['categories']:
        if cat['name'] in merge_names:
            old_ids.append(cat['id'])

    # 3. 更新 annotations：替换 category_id
    valid_image_ids = set()
    for ann in data['annotations']:
        if ann['category_id'] in old_ids:
            ann['category_id'] = target_id  # 指向目标 ID
            valid_image_ids.add(ann['image_id'])

    # 4. 清理 categories：删除被合并的旧类别
    data['categories'] = [cat for cat in data['categories'] if cat['name'] not in merge_names]

    # 5. 过滤无标注的图像


    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
