import pickle
import numpy as np
import torch
from pycocotools.coco import COCO
import cv2
import os
from collections import defaultdict


# --- 路径和配置 ---
GNN_RESULTS_PATH = 'gnn_corrected_results.pkl'
PKL_FILE_PATH = 'teeth_results_ap.pkl'
YOLO_JSON_PATH = 'best_predictions.json'

ANNOTATION_FILE_PATH = r'data/dataset/coco/crop_child/annotations/test_n.json'


IMAGE_ROOT_DIR = r'data/dataset/coco/crop_child/preprocessing_images'

# 用于保存可视化结果的输出文件夹
#OUTPUT_VIS_DIR = 'analysis_visualization_results'
OUTPUT_VIS_DIR = 'analysis_comparison_visualization'

CLASS_NAMES = [
    '11', '12', '13', '14', '15', '16', '17',
    '21', '22', '23', '24', '25', '26', '27',
    '31', '32', '33', '34', '35', '36', '37',
    '41', '42', '43', '44', '45', '46', '47',
    '51', '52', '53', '54', '55',
    '61', '62', '63', '64', '65',
    '71', '72', '73', '74', '75',
    '81', '82', '83', '84', '85'
]

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.3
epsilon = 1e-6


COLOR_TP = (0, 255, 0)  # 绿色 (True Positive)
COLOR_FN = (255, 0, 0)  # 蓝色 (False Negative - 漏检)
COLOR_FP_CLASS = (0, 0, 255)  # 红色 (False Positive - 编号错误)
COLOR_FP_HALLU = (0, 100, 255)  # 亮红色/暗橙 (False Positive - 无中生有)
COLOR_DUP = (0, 165, 255)  # 橙色 (Duplicate - 重复检测)


def calculate_iou(boxA, boxB):
    """计算两个边界框的交并比 (IoU)"""
    # 确保框是 [x1, y1, x2, y2] 格式
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + epsilon)
    return iou



def evaluate_predictions_for_image(pred_list, gt_anns, coco_id_to_name, model_idx_to_name):
    """
    对单张图片的预测列表进行评估，返回详细的错误分类。
    Args:
        pred_list (list): 包含 {'bbox': ..., 'score': ..., 'label_idx': ...} 的列表
        gt_anns (list): COCO格式的真实标注列表
    Returns:
        dict: 包含TP, FN, FP_CLASS等详细信息的字典
    """
    # 准备GT数据
    gt_bboxes = np.array([ann['bbox'] for ann in gt_anns])
    gt_labels_coco_id = np.array([ann['category_id'] for ann in gt_anns])
    if gt_bboxes.shape[0] > 0:
        gt_bboxes[:, 2] += gt_bboxes[:, 0]
        gt_bboxes[:, 3] += gt_bboxes[:, 1]

    # 准备预测数据
    pred_bboxes = np.array([p['bbox'] for p in pred_list])
    pred_scores = np.array([p['score'] for p in pred_list])
    pred_labels_model_idx = np.array([p['label_idx'] for p in pred_list])

    # 初始化状态追踪
    gt_matched = np.zeros(gt_bboxes.shape[0], dtype=bool)

    # 初始化结果存储
    results = {
        'TPs': [], 'FNs': [], 'FP_CLASS': [], 'FP_HALLU': [], 'DUPs': [],
        'confusion_pairs': defaultdict(int)
    }

    if len(pred_bboxes) > 0:
        sorted_pred_indices = np.argsort(pred_scores)[::-1]
    else:
        sorted_pred_indices = []

    # --- "三步判定法" ---
    for p_idx in sorted_pred_indices:
        pred_box = pred_bboxes[p_idx]
        pred_label_name = model_idx_to_name[pred_labels_model_idx[p_idx]]
        pred_score = pred_scores[p_idx]

        if gt_bboxes.shape[0] == 0:
            results['FP_HALLU'].append({'p_box': pred_box, 'pred_label': pred_label_name, 'score': pred_score})
            continue

        ious = np.array([calculate_iou(pred_box, gt_box) for gt_box in gt_bboxes])
        best_gt_idx = np.argmax(ious)
        max_iou = ious[best_gt_idx]

        gt_label_name = coco_id_to_name[gt_labels_coco_id[best_gt_idx]]

        if max_iou < IOU_THRESHOLD:
            results['FP_HALLU'].append({'p_box': pred_box, 'pred_label': pred_label_name, 'score': pred_score})
        else:
            if pred_label_name == gt_label_name:
                if not gt_matched[best_gt_idx]:
                    results['TPs'].append({'p_box': pred_box, 'label': pred_label_name, 'score': pred_score})
                    gt_matched[best_gt_idx] = True
                else:
                    results['DUPs'].append({'p_box': pred_box, 'pred_label': pred_label_name, 'score': pred_score})
            else:
                results['FP_CLASS'].append(
                    {'p_box': pred_box, 'pred_label': pred_label_name, 'gt_label': gt_label_name, 'score': pred_score})
                results['confusion_pairs'][f"(Pred:{pred_label_name} -> GT:{gt_label_name})"] += 1

    # 收集漏检 (FN)
    for gt_idx, is_matched in enumerate(gt_matched):
        if not is_matched:
            gt_label_name = coco_id_to_name[gt_labels_coco_id[gt_idx]]
            results['FNs'].append({'gt_box': gt_bboxes[gt_idx], 'gt_label': gt_label_name})

    return results


# --- 打印报告的函数 ---
def print_report(title, stats):
    print("\n\n" + "=" * 50)
    print(f"--- {title} ---")
    print("=" * 50)

    total_tp = stats['total_TP']
    total_dup = stats['total_DUP']
    total_fn = stats['total_FN']
    total_fp_class = stats['total_FP_CLASS']
    total_fp_hallu = stats['total_FP_HALLU']
    total_real_fps = total_fp_class + total_fp_hallu
    total_gts = total_tp + total_fn

    print(f"总计真阳性 (TPs): {total_tp}")
    print(f"总计漏检 (FNs): {total_fn}")
    print(f"总计有效真实标注 (GTs): {total_gts}")
    print("-" * 20)
    print(f"总计重复检测 (DUPs): {total_dup}")
    print(f"总计真实假阳性 (Real FPs): {total_real_fps}")
    print(f"   |-- 编号错误 (FP-Class): {total_fp_class}")
    print(f"   |-- 无中生有 (FP-Hallucination): {total_fp_hallu}")
    print("=" * 50)

    # 指标一: FP 构成
    percent_fp_class = total_fp_class / (total_real_fps + epsilon)
    print("\n--- 指标一: FP 构成比例 ---")
    print(f"  -> '编号错误' 占比: {percent_fp_class:.2%}")
    print(f"  -> '无中生有' 占比: {1 - percent_fp_class:.2%}")

    # 指标二: 临床编号准确率
    numbering_accuracy = total_tp / (total_tp + total_fp_class + epsilon)
    print("\n--- 指标二: 临床编号准确率 ---")
    print(f"  编号准确率: {numbering_accuracy:.4f}  (或 {numbering_accuracy:.2%})")

    # 指标三: 宏观性能
    overall_precision = total_tp / (total_tp + total_real_fps + total_dup + epsilon)
    overall_recall = total_tp / (total_gts + epsilon)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + epsilon)
    print("\n--- 指标三: 宏观性能指标 ---")
    print(f"  宏观精确度 (Precision): {overall_precision:.4f}")
    print(f"  宏观召回率 (Recall): {overall_recall:.4f}")
    print(f"  宏观 F1 分数 (F1-Score): {overall_f1:.4f}")
    print("=" * 50)


def draw_box(image, box, label, color, thickness=2):
    """在图像上绘制一个带标签的边界框"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # 设置标签文本
    font_scale = 0.5  # 调小一点字号以容纳更多信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
    text_w, text_h = text_size

    # 绘制文本背景框
    cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
    # 绘制文本
    cv2.putText(image, label, (x1, y1 - 3), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def main():
    print("正在加载文件...")
    gnn_results_tensor_keys = torch.load(GNN_RESULTS_PATH,map_location=lambda storage, loc: storage,weights_only=False)
    coco_gt = COCO(ANNOTATION_FILE_PATH)
    print("文件加载完成。")

    print("正在修正 GNN 结果字典的键类型...")
    gnn_results = {
        key.item() if isinstance(key, torch.Tensor) else int(key): value
        for key, value in gnn_results_tensor_keys.items()
    }
    print("键类型修正完成。")

    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    print(f"对比可视化结果将保存到: {os.path.abspath(OUTPUT_VIS_DIR)}")

    img_id_to_filename = {img_id: info['file_name'] for img_id, info in coco_gt.imgs.items()}

    # --- 准备标签映射 ---
    cat_ids = coco_gt.getCatIds(CLASS_NAMES)
    cats = coco_gt.loadCats(cat_ids)
    coco_id_to_name = {cat['id']: cat['name'] for cat in cats}
    model_idx_to_name = {i: name for i, name in enumerate(CLASS_NAMES)}
    print('标签映射创建完成....')

    # --- [修改] 初始化两个全局统计字典 ---
    global_stats_before = defaultdict(int)
    global_stats_after = defaultdict(int)
    global_stats_before['confusion_pairs'] = defaultdict(int)
    global_stats_after['confusion_pairs'] = defaultdict(int)

    img_ids_in_gt = sorted(coco_gt.getImgIds())
    print(f"\n开始分析 {len(img_ids_in_gt)} 张图片...")

    for i, img_id in enumerate(img_ids_in_gt):
        if img_id not in gnn_results:
            print("1")
            continue


        # --- 1. 准备数据 ---
        img_info = gnn_results[img_id]
        filename = os.path.basename(img_info['img_path'])

        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)

        # --- 2. 创建 "Before" 和 "After" 两个预测列表 ---
        preds_before = []
        preds_after = []

        for p in img_info['predictions']:
            preds_before.append({
                'bbox': p['bbox'],
                'score': p['original_score'],
                'label_idx': p['original_label_idx']
            })
            preds_after.append({
                'bbox': p['bbox'],
                'score': p['original_score'],  # 使用原始分数进行排序
                'label_idx': p['corrected_label_idx']
            })

        # --- 3. 分别评估 "Before" 和 "After" ---
        #results_train = evaluate_predictions_for_image(preds,gt_anns,coco_id_to_name, model_idx_to_name)
        results_before = evaluate_predictions_for_image(preds_before, gt_anns, coco_id_to_name, model_idx_to_name)
        results_after = evaluate_predictions_for_image(preds_after, gt_anns, coco_id_to_name, model_idx_to_name)

        # --- 4. 累加全局统计数据 ---
        for key in ['TPs', 'FNs', 'DUPs', 'FP_CLASS', 'FP_HALLU']:
            global_stats_before[f'total_{key[:-1]}'] += len(results_before[key])
            global_stats_after[f'total_{key[:-1]}'] += len(results_after[key])
        for pair, count in results_before['confusion_pairs'].items():
            global_stats_before['confusion_pairs'][pair] += count
        for pair, count in results_after['confusion_pairs'].items():
            global_stats_after['confusion_pairs'][pair] += count

        # --- 5. 拼接并保存可视化图像 ---
        image = cv2.imread(os.path.join(IMAGE_ROOT_DIR, filename))
        if image is None:
            continue

        vis_image_before = image.copy()
        vis_image_after = image.copy()

        # 绘制 "Before" 图像的错误
        for fn in results_before['FNs']: draw_box(vis_image_before, fn['gt_box'], f"MISS: {fn['gt_label']}", COLOR_FN)
        for fp in results_before['FP_CLASS']: draw_box(vis_image_before, fp['p_box'],
                                                       f"FP-C: P:{fp['pred_label']} (is {fp['gt_label']})",
                                                       COLOR_FP_CLASS)
        for fp in results_before['FP_HALLU']: draw_box(vis_image_before, fp['p_box'], f"FP-H: {fp['pred_label']}",
                                                       COLOR_FP_HALLU)
        for dup in results_before['DUPs']: draw_box(vis_image_before, dup['p_box'], f"DUP: {dup['pred_label']}",
                                                    COLOR_DUP)

        # 绘制 "After" 图像的错误
        for fn in results_after['FNs']: draw_box(vis_image_after, fn['gt_box'], f"MISS: {fn['gt_label']}", COLOR_FN)
        for fp in results_after['FP_CLASS']: draw_box(vis_image_after, fp['p_box'],
                                                      f"FP-C: P:{fp['pred_label']} (is {fp['gt_label']})",
                                                      COLOR_FP_CLASS)
        for fp in results_after['FP_HALLU']: draw_box(vis_image_after, fp['p_box'], f"FP-H: {fp['pred_label']}",
                                                      COLOR_FP_HALLU)
        for dup in results_after['DUPs']: draw_box(vis_image_after, dup['p_box'], f"DUP: {dup['pred_label']}",
                                                   COLOR_DUP)

        # 添加标签
        cv2.putText(vis_image_before, 'Before GNN (Faster R-CNN)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(vis_image_after, 'After GNN Correction', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 拼接图像
        comparison_image = cv2.hconcat([vis_image_before, vis_image_after])

        # 保存图像
        output_filepath = os.path.join(OUTPUT_VIS_DIR, filename)
        cv2.imwrite(output_filepath, comparison_image)

    # --- 6. 打印最终的对比报告 ---
    print_report("最终评估报告: Faster R-CNN 基线 (Before GNN)", global_stats_before)
    print_report("最终评估报告: GNN 修正后 (After GNN)", global_stats_after)


if __name__ == '__main__':
    main()