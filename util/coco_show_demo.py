from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os

dataset_dir = '../data/dataset/coco/crop_child/'

ann_file_path = os.path.join(dataset_dir,'annotations' ,'annotations.json')

coco = COCO(ann_file_path)


catIds = coco.getCatIds()
cats = coco.loadCats(catIds)

for cat in cats:
    print(f"cat:{cat}")


print("类别列表:", [cat['name'] for cat in coco.dataset['categories']])
print("标注数量:", len(coco.dataset['annotations']))
print("图像数量:", len(coco.dataset['images']))

img_id = coco.getImgIds()[10]
img_info = coco.loadImgs([img_id])[0]
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)
print(f"Image ID: {img_id}")
print(f"img_info: {img_info}")
print(f"Number of annotations: {len(anns)}")
print(f"ann_ids: {anns}")

# 可视化
image_path = os.path.join(dataset_dir,'preprocessing_images',img_info['file_name'])
image = plt.imread(image_path)
plt.imshow(image)
coco.showAnns(anns)  # 自动绘制框和分割掩码
plt.axis('off')
plt.show()
