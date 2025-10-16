import json


def delete_category(coco_path, output_path, delete_names,dsy_names):
    """
    :param merge_names: 待合并的类别名称列表（如 ["cat", "dog"]）
    :param target_name: 目标类别的名称（如 "animal"）
    """
    with open(coco_path, 'r') as f:
        data = json.load(f)

    # 1. 获取目标类别的 ID
    remove_categories_id = set()
    remove_categories_id.add(53)
    remove_categories_id.add(54)
    for cat in data['categories']:
        if cat['name'] in delete_names:
            remove_categories_id.add(cat['id'])

    print(remove_categories_id)

    if not remove_categories_id:
        raise ValueError(f"删除目标{delete_names} 不存在！")

    images_remove_set = set()
    for ann in data['annotations']:
        if ann['category_id'] in remove_categories_id:
            images_remove_set.add(ann['image_id'])
            print(images_remove_set)

    print(f"找到 {len(images_remove_set)} 张包含这些类别的图像")

    keep_images = []
    for img in data['images']:
        if img['file_name'].startswith('dsy'):
            print(f'移除多生牙:{img['file_name']}')
            images_remove_set.add(img['id'])

        if img['id'] not in images_remove_set:
            keep_images.append(img)



    keep_annotations = []
    for ann in data['annotations']:
        if ann['image_id'] not in images_remove_set:
            keep_annotations.append(ann)

    # 清理类别
    keep_cats = []
    for cat in data['categories']:
        if cat['name'] in delete_names or cat['name'] in dsy_names:
            print(f'跳过：{cat['name']}')
            continue
        keep_cats.append(cat)



    new_coco_data = {
        'info' : data.get('info',{}),
        "licenses": data.get('licenses', []),
        "images": keep_images,
        "annotations": keep_annotations,
        "categories": keep_cats
    }


    with open(output_path, 'w') as f:
        json.dump(new_coco_data, f, indent=4)
