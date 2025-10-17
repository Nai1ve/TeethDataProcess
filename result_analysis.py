import pickle
import numpy as np
from pycocotools.coco import COCO


PKL_FILE_PATH = 'raw_newdata.pkl'
ANNOTATION_FILE_PATH = r'data/dataset/coco/crop_child/annotations/test_n.json'
CLASS_NAMES = [
            '11','12','13','14','15','16','17',
            '21','22','23','24','25','26','27',
            '31','32','33','34','35','36','37',
            '41','42','43','44','45','46','47',
            '51','52','53','54','55',
            '61','62','63','64','65',
            '71','72','73','74','75',
            '81','82','83','84','85'
            ]

IOU_THRESHOLD = 0.5 # IOU 阈值
SCORE_THRESHOLD = 0.3# 置信度阈值

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou




def main():
    print("正在加载文件...")
    with open(PKL_FILE_PATH, 'rb') as f:
        results_list = pickle.load(f)
    coco_gt = COCO(ANNOTATION_FILE_PATH)
    print("文件加载完成。")

    print("正在创建类别数组索引和coco类别id的索引....")
    cat_ids = coco_gt.getCatIds(CLASS_NAMES)
    cats = coco_gt.loadCats(cat_ids)
    cat_name_to_id = {cat['name'] : cat['id'] for cat in cats}
    model_idx_to_coco_id = {i : cat_name_to_id[name] for i,name in enumerate(CLASS_NAMES)}
    coco_id_to_model_idx = {v:k for k,v in model_idx_to_coco_id.items()}
    print('标签映射创建完成....')

    per_image_stats = {}
    per_class_stats = {
        cls_id: {'TP': 0, 'FP': 0, 'FN': 0, 'iou_sum': 0, 'tp_count': 0}
        for cls_id in cat_ids
    }

    total_stats = {'TP': 0, 'FP': 0, 'FN': 0, 'DUP': 0}

    img_ids_in_gt = sorted(coco_gt.getImgIds())
    print(f"\n开始分析 {len(img_ids_in_gt)} 张图片...")

    # 逐张图片的处理
    for i, img_id in enumerate(img_ids_in_gt):

        per_image_stats[img_id] = {'FP': [], 'FN': [],'DUP':[], 'TP': 0, 'filename': coco_gt.loadImgs(img_id)[0]['file_name']}
        pred_result = results_list[i]

        assert pred_result['img_id'] == img_id, f"图片ID不匹配: GT={img_id}, Pred={pred_result['img_id']}"

        #获取对类别的遮罩
        pred_instances = pred_result['pred_instances']
        score_mask = pred_instances['scores'] >= SCORE_THRESHOLD
        pred_bboxes = pred_instances['bboxes'][score_mask].cpu().numpy()
        pred_labels_model_idx = pred_instances['labels'][score_mask].cpu().numpy()

        # 图片恢复
        # scale_factor = pred_result['scale_factor']
        # scale_factor_array = np.tile(scale_factor,2)
        # pred_bboxes_orig_scale = pred_bboxes / scale_factor_array

        # 获取真实标注
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)
        gt_bboxes = np.array([ann['bbox'] for ann in gt_anns if not ann.get('iscrowd', False)])
        gt_labels_coco_id = np.array([ann['category_id'] for ann in gt_anns if not ann.get('iscrowd', False)])

        #变更成【x1，y1，x2，y2】的格式方便计算
        if gt_bboxes.shape[0] > 0:
            gt_bboxes[:,2] += gt_bboxes[:,0]
            gt_bboxes[:,3] += gt_bboxes[:,1]

        # 记录真实框匹配的
        gt_matched = np.zeros(gt_bboxes.shape[0],dtype=bool)
        # 正确检验的数量
        num_tp_on_img = 0

        # 按照每类来处理
        for cls_idx,cls_coco_id in model_idx_to_coco_id.items():
            # 该类型下预测和真实值的遮罩
            pred_mask_cls = pred_labels_model_idx == cls_idx
            gt_mask_cls = gt_labels_coco_id == cls_coco_id

            # 获取原始尺度下的坐标
            pred_bboxes_cls = pred_bboxes[pred_mask_cls]
            # 获取真实的坐标
            gt_bboxes_cls = gt_bboxes[gt_mask_cls]

            if pred_bboxes_cls.shape[0] == 0 or gt_bboxes_cls.shape[0] == 0:
                continue

            # 记录IOU值的矩阵
            iou_matrix = np.zeros((pred_bboxes_cls.shape[0],gt_bboxes_cls.shape[0]))
            for p_idx,p_box in enumerate(pred_bboxes_cls):
                for g_idx,g_box in enumerate(gt_bboxes_cls):
                    iou_matrix[p_idx,g_idx] = calculate_iou(p_box,g_box)

            gt_indices_cls = np.where(gt_mask_cls)[0]
            pred_matched_cls = np.zeros(pred_bboxes_cls.shape[0], dtype=bool)

            for gt_idx in range(gt_bboxes_cls.shape[0]):
                best_match_iou = -1
                best_match_p_idx = -1
                for p_idx in range(pred_bboxes_cls.shape[0]):
                    if not pred_matched_cls[p_idx] and iou_matrix[p_idx,g_idx] > IOU_THRESHOLD:
                        if iou_matrix[p_idx,g_idx] > best_match_iou:
                            best_match_iou = iou_matrix[p_idx,g_idx]
                            best_match_p_idx = p_idx


                if best_match_p_idx != -1:
                    gt_matched[gt_indices_cls[g_idx]] = True
                    pred_matched_cls[best_match_p_idx] = True
                    per_class_stats[cls_coco_id]['TP'] += 1
                    per_class_stats[cls_coco_id]['iou_sum'] += best_match_iou
                    per_class_stats[cls_coco_id]['tp_count'] += 1
                    num_tp_on_img += 1

        num_fp_on_img = len(pred_bboxes) - num_tp_on_img
        num_fn_on_img = np.sum(~gt_matched)
        per_image_stats[img_id].update({'TP': num_tp_on_img, 'FP': num_fp_on_img, 'FN': num_fn_on_img})

    print("\n---- ")
    print("\n--- 分析完成，正在计算最终指标 ---")
    total_tp, total_fp, total_fn = 0, 0, 0
    epsilon = 1e-6

    for cls_coco_id in cat_ids:

        ann_ids_cls = coco_gt.getAnnIds(catIds=[cls_coco_id])
        num_gt_cls = len(coco_gt.loadAnns(ann_ids_cls))
        per_class_stats[cls_coco_id]['FN'] = num_gt_cls - per_class_stats[cls_coco_id]['TP']

        num_pred_cls = 0
        cls_model_idx = coco_id_to_model_idx[cls_coco_id]
        for res in results_list:
            score_mask = res['pred_instances']['scores'] >= SCORE_THRESHOLD
            labels = res['pred_instances']['labels'][score_mask].cpu().numpy()
            num_pred_cls += np.sum(labels == cls_model_idx)
        per_class_stats[cls_coco_id]['FP'] = num_pred_cls - per_class_stats[cls_coco_id]['TP']
        total_tp += per_class_stats[cls_coco_id]['TP']
        total_fp += per_class_stats[cls_coco_id]['FP']
        total_fn += per_class_stats[cls_coco_id]['FN']

    overall_precision = total_tp / (total_tp + total_fp + epsilon)
    overall_recall = total_tp / (total_tp + total_fn + epsilon)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + epsilon)
    overall_loc_accuracy = sum(s['iou_sum'] for s in per_class_stats.values()) / (
                sum(s['tp_count'] for s in per_class_stats.values()) + epsilon)

    print("\n--- 整体性能指标 ---")
    print(f"置信度阈值: {SCORE_THRESHOLD}, IoU 阈值: {IOU_THRESHOLD}")
    print(f"总计 True Positives (TP): {total_tp}")
    print(f"总计 False Positives (FP - 错误检测): {total_fp}")
    print(f"总计 False Negatives (FN - 漏检): {total_fn}")
    print("-" * 20)
    print(f"整体精确度 (Precision): {overall_precision:.4f}")
    print(f"整体召回率 (Recall / 敏感度): {overall_recall:.4f}")
    print(f"整体 F1 分数 (F1-Score): {overall_f1:.4f}")
    print(f"整体定位精度 (所有TP的平均IoU): {overall_loc_accuracy:.4f}")

    print("\n--- 分类别详细性能指标 ---")
    print("-" * 95)
    print(
        f"{'类别名称':<20} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'精确度':>8} | {'召回率':>8} | {'F1分数':>8} | {'平均IoU (定位精度)':>20}")
    print("-" * 95)

    for cls_coco_id, stats in per_class_stats.items():
        class_name = coco_gt.loadCats(cls_coco_id)[0]['name']
        tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        loc_accuracy = stats['iou_sum'] / (stats['tp_count'] + epsilon)
        print(f"{class_name:<20} | {tp:>5} | {fp:>5} | {fn:>5} | {precision:>8.3f} | {recall:>8.3f} | {f1_score:>8.3f} | {loc_accuracy:>20.3f}")
    print("-" * 95)

    print("\n--- 每张图片的漏检与错误检测情况 (只显示有问题的前20张) ---")
    sorted_images = sorted(per_image_stats.items(), key=lambda item: item[1]['FP'] + item[1]['FN'], reverse=True)
    for img_id, stats in sorted_images:
        if stats['FP'] > 0 or stats['FN'] > 0:
            print(f"图片: {stats['filename']} (ID: {img_id}) -> 漏检(FN): {stats['FN']}, 错检(FP): {stats['FP']}")

if __name__ == '__main__':
    main()