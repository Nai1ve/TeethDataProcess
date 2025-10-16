import os
import glob
import json
import logging
from collections import defaultdict
import cv2
import numpy as np
import torch

# ==============================================================================
# --- [ 1. 配置区 ] ---
# 你必须修改以下四个路径变量
# ==============================================================================

# 指向你的YOLO预测结果JSON文件 (COCO格式)
YOLO_PREDICTION_JSON_PATH = 'best_predictions.json'

# 指向包含YOLO格式 .txt 真实标注的文件夹
YOLO_GT_TXT_DIR = r'data/dataset/yolo/process/labels/test/'

# 指向包含测试集原始图片的文件夹 (用于获取尺寸和进行可视化)
TEST_IMAGE_DIR = r'data/dataset/yolo/process/images/test/'

# 用于保存可视化结果的输出文件夹
OUTPUT_VIS_DIR = 'analysis_yolo_visualization'

# --- 其他配置 ---
CLASS_NAMES = [  # 48个类别，请确保顺序与你的模型训练时一致
    '11', '12', '13', '14', '15', '16', '17', '21', '22', '23', '24', '25', '26', '27',
    '31', '32', '33', '34', '35', '36', '37', '41', '42', '43', '44', '45', '46', '47',
    '51', '52', '53', '54', '55', '61', '62', '63', '64', '65', '71', '72', '73', '74', '75',
    '81', '82', '83', '84', '85'
]
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.3
epsilon = 1e-6

# --- 可视化颜色 (BGR格式) ---
COLOR_TP = (0, 255, 0)
COLOR_FN = (255, 0, 0)
COLOR_FP_CLASS = (0, 0, 255)
COLOR_FP_HALLU = (0, 100, 255)
COLOR_DUP = (0, 165, 255)


# ==============================================================================
# --- [ 2. 辅助函数区 ] ---
# ==============================================================================

def calculate_iou(boxA, boxB):
    """计算两个边界框 [x1, y1, x2, y2] 的交并比 (IoU)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + epsilon)
    return iou


def draw_box(image, box, label, color, thickness=2):
    """在图像上绘制一个带标签的边界框"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
    text_w, text_h = text_size
    cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 3), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def load_yolo_gt_from_txts(gt_txt_dir, image_dir):
    """从包含YOLO格式.txt文件的文件夹中加载所有真实标注。"""
    print(f"正在从TXT文件夹加载YOLO真实标注: {gt_txt_dir}")
    gt_data_dict = {}
    txt_files = glob.glob(os.path.join(gt_txt_dir, '*.txt'))

    for txt_path in txt_files:
        filename_no_ext = os.path.splitext(os.path.basename(txt_path))[0]

        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            potential_path = os.path.join(image_dir, filename_no_ext + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path is None: continue

        image = cv2.imread(img_path)
        img_h, img_w, _ = image.shape

        gt_bboxes_img, gt_labels_img = [], []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue

                class_idx, x_c, y_c, w, h = map(float, parts)

                abs_w, abs_h = w * img_w, h * img_h
                x1 = (x_c * img_w) - abs_w / 2
                y1 = (y_c * img_h) - abs_h / 2
                x2, y2 = x1 + abs_w, y1 + abs_h

                gt_bboxes_img.append([x1, y1, x2, y2])
                gt_labels_img.append(int(class_idx))

        image_filename = os.path.basename(img_path)
        gt_data_dict[image_filename] = {
            'bboxes': np.array(gt_bboxes_img, dtype=np.float32),
            'labels': np.array(gt_labels_img, dtype=np.int64)
        }

    print(f"YOLO真实标注加载完成，共处理 {len(gt_data_dict)} 张图片。")
    return gt_data_dict


def load_yolo_predictions_from_json(json_path):
    """加载YOLO JSON预测结果, 假设'image_id'是文件名。"""
    print(f"正在加载并解析YOLO JSON预测文件: {json_path}")
    with open(json_path, 'r') as f:
        yolo_preds = json.load(f)

    preds_by_filename = defaultdict(lambda: {'scores': [], 'labels': [], 'bboxes': []})
    for pred in yolo_preds:
        if pred['score'] < SCORE_THRESHOLD: continue

        # 假设 image_id 就是 filename, e.g., "001.jpg"
        filename = str(pred['image_id']) + '.jpg'

        preds_by_filename[filename]['scores'].append(pred['score'])
        preds_by_filename[filename]['labels'].append(pred['category_id'])
        x, y, w, h = pred['bbox']
        preds_by_filename[filename]['bboxes'].append([x, y, x + w, y + h])

    # 将列表转换为numpy数组
    for filename in preds_by_filename:
        preds_by_filename[filename]['scores'] = np.array(preds_by_filename[filename]['scores'], dtype=np.float32)
        preds_by_filename[filename]['labels'] = np.array(preds_by_filename[filename]['labels'], dtype=np.int64)
        preds_by_filename[filename]['bboxes'] = np.array(preds_by_filename[filename]['bboxes'], dtype=np.float32)

    print("YOLO JSON文件解析完成。")
    return preds_by_filename


# ==============================================================================
# --- [ 3. 主函数区 ] ---
# ==============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    # 1. 加载所有数据
    gt_data_dict = load_yolo_gt_from_txts(YOLO_GT_TXT_DIR, TEST_IMAGE_DIR)
    pred_data_dict = load_yolo_predictions_from_json(YOLO_PREDICTION_JSON_PATH)

    sorted_filenames = sorted(gt_data_dict.keys())

    # 2. 初始化全局统计
    global_stats = defaultdict(int)
    global_stats['confusion_pairs'] = defaultdict(int)

    print(f"\n开始分析 {len(sorted_filenames)} 张图片...")

    # 3. 主分析循环
    for filename in sorted_filenames:
        # 获取GT
        current_gt = gt_data_dict[filename]
        gt_bboxes = current_gt['bboxes']
        gt_labels = current_gt['labels']
        global_stats['total_GT'] += len(gt_bboxes)

        # 获取预测 (使用.get安全处理没有预测结果的图片)
        default_pred = {'bboxes': np.array([]), 'labels': np.array([]), 'scores': np.array([])}
        current_pred = pred_data_dict.get(filename, default_pred)
        pred_bboxes = current_pred['bboxes']
        pred_scores = current_pred['scores']
        pred_labels = current_pred['labels']

        # --- 核心匹配逻辑 ---
        gt_matched = np.zeros(gt_bboxes.shape[0], dtype=bool)

        if len(pred_bboxes) > 0:
            sorted_pred_indices = np.argsort(pred_scores)[::-1]
        else:
            sorted_pred_indices = []

        # (可视化) 初始化图像
        img_path = os.path.join(TEST_IMAGE_DIR, filename)
        vis_image = cv2.imread(img_path)

        for p_idx in sorted_pred_indices:
            pred_box = pred_bboxes[p_idx]
            pred_label_idx = pred_labels[p_idx]
            pred_label_name = CLASS_NAMES[pred_label_idx]
            pred_score = pred_scores[p_idx]

            status = 'UNACCOUNTED'

            if gt_bboxes.shape[0] == 0:
                status = 'FP_HALLU'
            else:
                ious = np.array([calculate_iou(pred_box, gt_box) for gt_box in gt_bboxes])
                best_gt_idx = np.argmax(ious)
                max_iou = ious[best_gt_idx]

                if max_iou < IOU_THRESHOLD:
                    status = 'FP_HALLU'
                else:
                    gt_label_idx = gt_labels[best_gt_idx]
                    if not gt_matched[best_gt_idx]:
                        if pred_label_idx == gt_label_idx:
                            status = 'TP'
                        else:
                            status = 'FP_CLASS'
                        gt_matched[best_gt_idx] = True
                    else:
                        status = 'DUP'

            # 更新全局统计并进行可视化
            global_stats[f'total_{status}'] += 1
            if status == 'FP_CLASS':
                gt_label_name = CLASS_NAMES[gt_labels[best_gt_idx]]
                pair = f"(Pred:{pred_label_name} -> GT:{gt_label_name})"
                global_stats['confusion_pairs'][pair] += 1
                if vis_image is not None:
                    label = f"FP-C: P {pred_label_name} (is {gt_label_name})"
                    draw_box(vis_image, pred_box, label, COLOR_FP_CLASS)
            elif status == 'FP_HALLU':
                if vis_image is not None:
                    label = f"FP-H: P {pred_label_name}"
                    draw_box(vis_image, pred_box, label, COLOR_FP_HALLU)
            elif status == 'DUP':
                if vis_image is not None:
                    label = f"DUP: P {pred_label_name}"
                    draw_box(vis_image, pred_box, label, COLOR_DUP)

        # 收集并绘制FN
        unmatched_gt_indices = np.where(~gt_matched)[0]
        global_stats['total_FN'] += len(unmatched_gt_indices)
        if vis_image is not None:
            for gt_idx in unmatched_gt_indices:
                gt_label_name = CLASS_NAMES[gt_labels[gt_idx]]
                draw_box(vis_image, gt_bboxes[gt_idx], f"MISS: {gt_label_name}", COLOR_FN)

            # 保存可视化图像
            output_filepath = os.path.join(OUTPUT_VIS_DIR, filename)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            cv2.imwrite(output_filepath, vis_image)

    # --- 4. 最终报告生成 ---
    print("\n\n" + "=" * 40)
    print("--- YOLOv5 核心错误构成分析 ---")
    print("=" * 40)
    total_real_fps = global_stats['total_FP_CLASS'] + global_stats['total_FP_HALLU']
    print(f"总计真阳性 (Total TPs): {global_stats['total_TP']}")
    print(f"总计漏检 (Total FNs): {global_stats['total_FN']}")
    print(f"总计有效真实标注 (Total GTs): {global_stats['total_GT']}")
    print("-" * 20)
    print(f"总计重复检测 (Total DUPs): {global_stats['total_DUP']}")
    print(f"总计真实假阳性 (Total Real FPs): {total_real_fps}")
    print(f"   |-- 编号错误 (FP-Classification): {global_stats['total_FP_CLASS']}")
    print(f"   |-- 无中生有 (FP-Hallucination): {global_stats['total_FP_HALLU']}")
    print("=" * 40)

    percent_fp_class = global_stats['total_FP_CLASS'] / (total_real_fps + epsilon)
    print("\n--- 指标一: FP 构成比例 ---")
    print(f"在所有 {total_real_fps} 个真实FP错误中:")
    print(f"  -> '编号错误' 占比: {percent_fp_class:.2%}")

    numbering_accuracy = global_stats['total_TP'] / (
                global_stats['total_TP'] + global_stats['total_FP_CLASS'] + epsilon)
    print("\n--- 指标二: 临床编号准确率 (Numbering Accuracy) ---")
    print(f"  编号准确率: {numbering_accuracy:.4f} (或 {numbering_accuracy:.2%})")

    overall_precision = global_stats['total_TP'] / (
                global_stats['total_TP'] + total_real_fps + global_stats['total_DUP'] + epsilon)
    overall_recall = global_stats['total_TP'] / (global_stats['total_GT'] + epsilon)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + epsilon)
    print("\n--- 指标三: 宏观性能指标 ---")
    print(f"  宏观精确度 (Overall Precision): {overall_precision:.4f}")
    print(f"  宏观召回率 (Overall Recall): {overall_recall:.4f}")
    print(f"  宏观 F1 分数 (Overall F1-Score): {overall_f1:.4f}")

    print("\n--- 指标四: 典型混淆对 ---")
    sorted_confusion = sorted(global_stats['confusion_pairs'].items(), key=lambda item: item[1], reverse=True)
    for pair, count in sorted_confusion[:10]:
        print(f"  {pair:<30} : {count} 次")
    print("=" * 40)


if __name__ == '__main__':
    main()