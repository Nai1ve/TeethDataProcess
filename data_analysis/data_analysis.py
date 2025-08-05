import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pprint
from pycocotools.coco import COCO

file_path = '../data/analysis/'

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

root_dir_path = '/Users/liupeize/Desktop/牙齿/儿童牙齿数据'

def raw_photo_count():
    """
    原始数据统计，总牙片数，废片数、多生牙、缺牙、困难、中等、简单、分辨率、医院、长款比
    """
    hospital_list = []
    for hospital in os.listdir(root_dir_path):
        if not hospital.startswith('.'):
            photo_dic = {
                'path' : root_dir_path,
                'hospital' : hospital
            }
            hospital_list.append(photo_dic)


    for hospital in hospital_list:
        photo_count = 0
        ann_count = 0
        easy = 0
        hard = 0
        mid = 0
        rejected_film = 0
        supernumerary_tooth = 0
        missing_tooth = 0
        wh_dic ={}
        hospital_dir = os.path.join(hospital['path'],hospital['hospital'])
        for root,sub_dir_names,file_names in os.walk(hospital_dir):
            for file_name in file_names:
                if not file_name.startswith('.'):

                    if file_name.endswith('.jpg'):
                        photo_count+=1

                        # 统计简单困难
                        type = root.split('/')[-1]
                        if type == '废片' or type == '废片（未标注）':
                            rejected_film += 1
                        elif type == '困难' or type == '复杂':
                            hard+=1
                        elif type == '中等':
                            mid+=1
                        elif type == '容易' or type == '简单':
                            easy+=1
                        elif type == '标记完成1-90' or type == '220已标记完' or type == '曲面断层片已标记1-14':
                            continue

                        elif type == '多生牙' or type =='缺牙' or type =='缺失牙':
                            if type == '多生牙':
                                supernumerary_tooth+=1
                            elif type == '缺牙' or type == '缺失牙':
                                missing_tooth+=1

                            type = root.split('/')[-2]
                            if type == '简单' or type == '容易':
                                easy+=1
                            elif type == '中等':
                                mid+=1
                            elif type == '复杂':
                                hard+=1
                        else:
                            print(f"error:{type}")
                            print(f"root:{root}")

                        # 统计多生牙缺牙
                        if file_name.startswith('dsy'):
                            supernumerary_tooth+=1
                        elif file_name.startswith('qy'):
                            missing_tooth+=1

                        # 统计图片宽高
                        img_path = os.path.join(root,file_name)
                        img = Image.open(img_path)
                        width,height = img.size
                        key = str(width)+ '*'+str(height)
                        value = wh_dic.get(key,[0])
                        wh_dic[key] = [value[0] + 1, img_path,width / height]

                    if file_name.endswith('.json'):
                        ann_count+=1

        hospital['photo_count'] = photo_count
        hospital['ann_count'] = ann_count
        hospital['rejected_film'] = rejected_film
        hospital['easy'] = easy
        hospital['hard'] = hard
        hospital['mid'] = mid
        hospital['missing_tooth'] = missing_tooth
        hospital['supernumerary_tooth'] = supernumerary_tooth
        hospital['wh_dic'] = wh_dic
        pprint.pprint(hospital)


    return hospital_list

child_teeth_dir = '../data/dataset/coco/crop_child'
ann_file_path = os.path.join(child_teeth_dir,'annotations','annotations.json')

def coco_teeth_count():
    """
    统计各类牙齿的数量，以及牙齿对牙片的尺寸，长宽比,FDI牙齿数量，尺寸，比例
    """
    coco = COCO(ann_file_path)

    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)

    cat_names = [cat['name'] for cat in cats]
    # cat_ids = [cat['id'] for cat in cats]


    category_count = {name:0 for name in cat_names}
    aspect_ratios = {name: [] for name in cat_names}
    image_ratios = {}
    image_annId_size = {}


    for img_id in coco.getImgIds():
        img = coco.loadImgs(img_id)[0]
        img_width,img_height = img['width'],img['height']


        # 获取当前图片的标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = ann['category_id']
            cat_name = coco.loadCats(cat_id)[0]['name']

            # 统计类别数量
            category_count[cat_name] += 1

            # 计算标注框长宽比（宽/高）
            bbox = ann['bbox']
            aspect_ratio = bbox[2] / bbox[3]
            aspect_ratios[cat_name].append(aspect_ratio)

            fdi_ratios = image_ratios.get(img_id,{})
            fdi_ratios_list = fdi_ratios.get(cat_name,[])
            fdi_ratios_list.append((bbox[2] * bbox[3]) / (img_width * img_height))
            fdi_ratios[cat_name] = fdi_ratios_list
            image_ratios[img_id] = fdi_ratios

            annId_size_dict = image_annId_size.get(img_id,{})
            size_list = annId_size_dict.get(cat_name,[])
            size_list.append(str(bbox[2]) + '*' + str(bbox[3]))
            annId_size_dict[cat_name] = size_list
            image_annId_size[img_id] =annId_size_dict

            if bbox[2] < 32 or bbox[3] < 32:
                print(img_id)


    pprint.pprint(cat_names)
    pprint.pprint(category_count)
    pprint.pprint(aspect_ratios)
    pprint.pprint(image_ratios)
    pprint.pprint(image_annId_size)








def teeth_count():
    """
    统计牙齿的数量
    """

    dic = {}

    for dir in os.listdir(file_path):
        dic[dir] = len(os.listdir(os.path.join(file_path,dir)))

    dic = dict(sorted(dic.items()))

    # print(f"dic key size:{len(dic.keys())}")
    # del dic['9']
    # del dic['91']
    # del dic['247']
    # del dic['325']
    # del dic['326']
    # del dic['432']
    # del dic['674']
    # print(f"dic key size:{len(dic.keys())}")

    categories = dic.keys()

    plt.figure(figsize=(20,8))
    plt.bar(categories,dic.values())
    plt.xticks(rotation=45,ha='right',fontsize=8)
    plt.subplots_adjust(bottom=0.3)

    plt.title("teeth count Chart")
    plt.xlabel("FDI")
    plt.ylabel("count")
    plt.tight_layout()

    plt.savefig("teeh_count_chart.png")
    plt.show()


#teeth_count()



def teeth_size():
    """
    统计牙齿的大小
    """

    for dir in os.listdir(file_path):
        teeth_dir = os.path.join(file_path,dir)
        print(teeth_dir)
        for dirpath, dirnames,filenames in os.walk(teeth_dir):
            print(f"当前目录路径：{dirpath}")
            print(f"子目录：{dirnames}")
            print(f"文件：{filenames}")

            for filename in filenames:
                img_path = os.path.join(dirpath,filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(x, y, gray, cmap='gray', rstride=5, cstride=5)
                ax.set_title("牙齿灰度3D分布图")
                ax.set_zlabel("亮度值")
                plt.savefig("3d_gray.png", dpi=300)  # 保存高清图
                plt.show()

                print(img.shape)

        print(f"{teeth_dir} finish")

# teeth_size()

if __name__ == '__main__':
    # raw_photo_count()
    coco_teeth_count()