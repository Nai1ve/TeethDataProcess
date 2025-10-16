import pickle
import numpy as np
from pycocotools.coco import COCO
import cv2
import os
from collections import defaultdict


# --- 路径和配置 ---
PKL_FILE_PATH = 'teeth_results_ap.pkl'
YOLO_JSON_PATH = 'best_predictions.json'

ANNOTATION_FILE_PATH = r'data/dataset/coco/crop_child/annotations/test.json'


IMAGE_ROOT_DIR = r'data/dataset/coco/crop_child/preprocessing_images'

# 用于保存可视化结果的输出文件夹
OUTPUT_VIS_DIR = 'analysis_visualization_results'

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
    with open(PKL_FILE_PATH, 'rb') as f:
        results_list = pickle.load(f)
    coco_gt = COCO(ANNOTATION_FILE_PATH)



    print("文件加载完成。")

    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    print(f"可视化结果将保存到: {os.path.abspath(OUTPUT_VIS_DIR)}")

    print("正在创建类别数组索引和coco类别id的索引....")
    cat_ids = coco_gt.getCatIds(CLASS_NAMES)
    cats = coco_gt.loadCats(cat_ids)
    cat_name_to_id = {cat['name']: cat['id'] for cat in cats}

    coco_id_to_name = {v: k for k, v in cat_name_to_id.items()}
    model_idx_to_name = {i: name for i, name in enumerate(CLASS_NAMES)}
    print('标签映射创建完成....')


    per_image_stats = {}


    global_stats = {
        'total_TP': 0,
        'total_FP_CLASS': 0,
        'total_FP_HALLU': 0,
        'total_FN': 0,
        'total_DUP': 0,
        'total_GT': 0,
        'confusion_pairs': defaultdict(int)  # 用于混淆矩阵
    }

    img_ids_in_gt = sorted(coco_gt.getImgIds())
    print(f"\n开始分析 {len(img_ids_in_gt)} 张图片...")

    # 逐张图片进行处理
    for i, img_id in enumerate(img_ids_in_gt):

        img_info = coco_gt.loadImgs(img_id)[0]
        filename = img_info['file_name']


        per_image_stats[img_id] = {
            'filename': filename,
            'TP_details': [],
            'FP_CLASS_details': [],
            'FP_HALLU_details': [],
            'FN_details': [],
            'DUP_details': []
        }

        img_path = os.path.join(IMAGE_ROOT_DIR, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"\n[警告] 无法加载图片: {img_path}. 将跳过此图片的可视化。")

        pred_result = results_list[i]
        assert pred_result['img_id'] == img_id, f"图片ID不匹配: GT={img_id}, Pred={pred_result['img_id']}"

        # 获取预测结果并根据置信度阈值进行过滤
        pred_instances = pred_result['pred_instances']
        score_mask = pred_instances['scores'] >= SCORE_THRESHOLD
        pred_bboxes = pred_instances['bboxes'][score_mask].cpu().numpy()
        pred_labels_model_idx = pred_instances['labels'][score_mask].cpu().numpy()
        pred_scores = pred_instances['scores'][score_mask].cpu().numpy()

        # 获取真实标注
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)

        valid_gt_anns = [ann for ann in gt_anns if
                         not ann.get('iscrowd', False) and ann['bbox'][2] > 0 and ann['bbox'][3] > 0]
        gt_bboxes = np.array([ann['bbox'] for ann in valid_gt_anns])
        gt_labels_coco_id = np.array([ann['category_id'] for ann in valid_gt_anns])

        # 累加GT总数
        global_stats['total_GT'] += len(gt_bboxes)

        if gt_bboxes.shape[0] > 0:
            gt_bboxes[:, 2] += gt_bboxes[:, 0]
            gt_bboxes[:, 3] += gt_bboxes[:, 1]


        gt_matched = np.zeros(gt_bboxes.shape[0], dtype=bool)
        pred_status = np.array(['UNACCOUNTED'] * len(pred_bboxes), dtype=object)

        if len(pred_bboxes) > 0:
            sorted_pred_indices = np.argsort(pred_scores)[::-1]
        else:
            sorted_pred_indices = []

        # 遍历所有排序后的预测框
        for p_idx in sorted_pred_indices:
            pred_box = pred_bboxes[p_idx]
            pred_label_model_idx = pred_labels_model_idx[p_idx]
            pred_label_name = model_idx_to_name[pred_label_model_idx]

            if gt_bboxes.shape[0] == 0:
                pred_status[p_idx] = 'FP_HALLU'
                continue

            # 1. 寻找最佳空间匹配
            ious = np.array([calculate_iou(pred_box, gt_box) for gt_box in gt_bboxes])
            best_gt_idx = np.argmax(ious)
            max_iou = ious[best_gt_idx]

            best_gt_coco_id = gt_labels_coco_id[best_gt_idx]
            if best_gt_coco_id in [54,53]:
                continue

            best_gt_label_name = coco_id_to_name[best_gt_coco_id]

            # 2. & 3. 判断位置和标签
            if max_iou < IOU_THRESHOLD:
                pred_status[p_idx] = 'FP_HALLU'
            else:
                if pred_label_name == best_gt_label_name:
                    if not gt_matched[best_gt_idx]:
                        pred_status[p_idx] = 'TP'
                        gt_matched[best_gt_idx] = True
                    else:
                        pred_status[p_idx] = 'DUP'
                else:
                    pred_status[p_idx] = 'FP_CLASS'



        # 4.1 收集 FN (漏检)
        unmatched_gt_indices = np.where(~gt_matched)[0]
        for gt_idx in unmatched_gt_indices:
            fn_box = gt_bboxes[gt_idx]
            fn_label_id = gt_labels_coco_id[gt_idx]
            if fn_label_id in [53,54]:
                continue
            fn_label_name = coco_id_to_name[fn_label_id]
            per_image_stats[img_id]['FN_details'].append({
                'gt_box': fn_box,
                'gt_label': fn_label_name
            })
            global_stats['total_FN'] += 1  # 累加到全局FN

        # 4.2 遍历所有预测框的状态，填充详情
        for p_idx, status in enumerate(pred_status):
            pred_box = pred_bboxes[p_idx]
            pred_label_name = model_idx_to_name[pred_labels_model_idx[p_idx]]
            score = pred_scores[p_idx]


            if gt_bboxes.shape[0] > 0:
                ious = np.array([calculate_iou(pred_box, gt_box) for gt_box in gt_bboxes])
                best_gt_idx = np.argmax(ious)
                max_iou = ious[best_gt_idx]
                if gt_labels_coco_id[best_gt_idx] in [53,54]:
                    continue
                best_gt_label_name = coco_id_to_name[gt_labels_coco_id[best_gt_idx]]
            else:
                max_iou = 0
                best_gt_label_name = "N/A"

            if status == 'TP':
                global_stats['total_TP'] += 1
                per_image_stats[img_id]['TP_details'].append({
                    'p_box': pred_box,
                    'gt_box': gt_bboxes[best_gt_idx],
                    'label_name': pred_label_name,
                    'score': score,
                    'iou': max_iou
                })

            elif status == 'DUP':
                global_stats['total_DUP'] += 1
                per_image_stats[img_id]['DUP_details'].append({
                    'p_box': pred_box,
                    'pred_label': pred_label_name,
                    'gt_match_label': best_gt_label_name,
                    'score': score
                })

            elif status == 'FP_CLASS':
                global_stats['total_FP_CLASS'] += 1
                global_stats['confusion_pairs'][f"(Pred:{pred_label_name} -> GT:{best_gt_label_name})"] += 1
                per_image_stats[img_id]['FP_CLASS_details'].append({
                    'p_box': pred_box,
                    'pred_label': pred_label_name,
                    'gt_match_label': best_gt_label_name,
                    'score': score,
                    'iou_with_gt': max_iou
                })

            elif status == 'FP_HALLU':
                global_stats['total_FP_HALLU'] += 1
                per_image_stats[img_id]['FP_HALLU_details'].append({
                    'p_box': pred_box,
                    'pred_label': pred_label_name,
                    'score': score,
                    'max_iou_any_gt': max_iou
                })


        if image is not None:
            vis_image = image.copy()
            stats_for_img = per_image_stats[img_id]

            # 1. 绘制 FN (蓝色)
            for fn in stats_for_img['FN_details']:
                label = f"MISS: {fn['gt_label']}"
                draw_box(vis_image, fn['gt_box'], label, COLOR_FN)

            # 2. 绘制 FP-CLASS (红色)
            for fp in stats_for_img['FP_CLASS_details']:
                label = f"FP-CLASS: PRED {fp['pred_label']} (IS {fp['gt_match_label']}) S:{fp['score']:.2f}"
                draw_box(vis_image, fp['p_box'], label, COLOR_FP_CLASS)

            # 3. 绘制 FP-HALLU (亮红色)
            for fp in stats_for_img['FP_HALLU_details']:
                label = f"FP-HALLU: {fp['pred_label']} S:{fp['score']:.2f}"
                draw_box(vis_image, fp['p_box'], label, COLOR_FP_HALLU)

            # 4. 绘制 DUP (橙色)
            for dup in stats_for_img['DUP_details']:
                label = f"DUP: {dup['pred_label']} S:{dup['score']:.2f}"
                draw_box(vis_image, dup['p_box'], label, COLOR_DUP)

            # 5. 绘制 TP (绿色)
            # for tp in stats_for_img['TP_details']:
            #     label = f"TP: {tp['label_name']} ({tp['score']:.2f})"
            #     draw_box(vis_image, tp['p_box'], label, COLOR_TP)

            # 保存图像
            output_filepath = os.path.join(OUTPUT_VIS_DIR, filename)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            cv2.imwrite(output_filepath, vis_image)



    print("\n\n" + "=" * 40)
    print("--- 核心错误构成分析 (Error Analysis Report) ---")
    print("=" * 40)

    total_tp = global_stats['total_TP']
    total_dup = global_stats['total_DUP']
    total_fn = global_stats['total_FN']
    total_fp_class = global_stats['total_FP_CLASS']
    total_fp_hallu = global_stats['total_FP_HALLU']
    total_real_fps = total_fp_class + total_fp_hallu
    total_gts = global_stats['total_GT']  # TPs + FNs

    # 验证计数器
    #assert total_gts == (total_tp + total_fn), f"GT计数错误! Total_GT:{total_gts} vs TP+FN:{total_tp + total_fn}"

    print(f"总计真阳性 (Total TPs): {total_tp}")
    print(f"总计漏检 (Total FNs): {total_fn}")
    print(f"总计有效真实标注 (Total GTs): {total_gts}")
    print("-" * 20)
    print(f"总计重复检测 (Total DUPs): {total_dup}")
    print(f"总计真实假阳性 (Total Real FPs): {total_real_fps}")
    print(f"   |-- 编号错误 (FP-Classification): {total_fp_class}")
    print(f"   |-- 无中生有 (FP-Hallucination): {total_fp_hallu}")
    print("=" * 40)

    # --- 指标一: FP 构成比例 (饼图数据) ---
    percent_fp_class = total_fp_class / (total_real_fps + epsilon)
    percent_fp_hallu = total_fp_hallu / (total_real_fps + epsilon)
    print("\n--- 指标一: FP 构成比例 ---")
    print(f"在所有 {total_real_fps} 个真实FP错误中:")
    print(f"  -> '编号错误' 占比: {percent_fp_class:.2%} ({total_fp_class} 个)")
    print(f"  -> '无中生有' 占比: {percent_fp_hallu:.2%} ({total_fp_hallu} 个)")

    # --- 指标二: 临床编号准确率  ---
    numbering_accuracy = total_tp / (total_tp + total_fp_class + epsilon)
    print("\n--- 指标二: 临床编号准确率 (Numbering Accuracy) ---")
    print(f"  定义: 在所有被框住的牙齿中(TP + FP-Class)，编号也正确的比例")
    print(f"  计算: Total_TPs / (Total_TPs + Total_FP_Class)")
    print(f"  编号准确率: {numbering_accuracy:.4f}  (或 {numbering_accuracy:.2%})")

    # --- 指标三: 宏观性能指标 (用于参考) ---
    overall_precision = total_tp / (total_tp + total_real_fps + total_dup + epsilon)
    overall_recall = total_tp / (total_gts + epsilon)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + epsilon)

    print("\n--- 指标三: 宏观性能指标 (Overall Metrics) ---")
    print(f"  (将 DUP 视为一种 FP 进行计算)")
    print(f"  宏观精确度 (Overall Precision): {overall_precision:.4f}")
    print(f"  宏观召回率 (Overall Recall): {overall_recall:.4f}")
    print(f"  宏观 F1 分数 (Overall F1-Score): {overall_f1:.4f}")

    # --- 指标四: 典型混淆对 ---
    print("\n--- 指标四: 典型混淆对 (Top 10 Confusion Pairs) ---")
    if not global_stats['confusion_pairs']:
        print("未发现编号错误 (FP-Classification errors)。")
    else:
        sorted_confusion = sorted(global_stats['confusion_pairs'].items(), key=lambda item: item[1], reverse=True)
        for pair, count in sorted_confusion[:10]:  # 打印前10个
            print(f"  {pair:<30} : {count} 次")
    print("=" * 40)


    print("\n--- 每张图片的详细错误日志 (按总错误数排序) ---")
    print(f"(详细可视化图像已保存在: {os.path.abspath(OUTPUT_VIS_DIR)})")

    sorted_images = sorted(per_image_stats.items(),
                           key=lambda item: len(item[1]['FP_CLASS_details']) + len(item[1]['FP_HALLU_details']) + len(
                               item[1]['FN_details']) + len(item[1]['DUP_details']),
                           reverse=True)

    for img_id, stats in sorted_images:
        total_errors = len(stats['FN_details']) + len(stats['FP_CLASS_details']) + len(stats['FP_HALLU_details']) + len(
            stats['DUP_details'])
        if total_errors == 0:
            continue

        print("\n" + "=" * 80)
        print(f"图片: {stats['filename']} (ID: {img_id})")
        print(
            f"    -> 摘要: 漏检(FN): {len(stats['FN_details'])}, 编号错误(FP-C): {len(stats['FP_CLASS_details'])}, 无中生有(FP-H): {len(stats['FP_HALLU_details'])}, 重复(DUP): {len(stats['DUP_details'])}")

        if stats['FN_details']:
            print("    [详细漏检 (FN)]: ")
            for fn in stats['FN_details']:
                print(f"      - [应检测] 类别: {fn['gt_label']:<5} @ 框: {[int(c) for c in fn['gt_box']]}")

        if stats['FP_CLASS_details']:
            print("    [详细编号错误 (FP-CLASS)]: ")
            for fp in stats['FP_CLASS_details']:
                print(
                    f"      - [错误预测] 类别: {fp['pred_label']:<5} (真实应为: {fp['gt_match_label']}), 置信度: {fp['score']:.3f}, IoU: {fp['iou_with_gt']:.2f}")

        if stats['FP_HALLU_details']:
            print("    [详细无中生有 (FP-HALLU)]: ")
            for fp in stats['FP_HALLU_details']:
                print(
                    f"      - [错误预测] 类别: {fp['pred_label']:<5}, 置信度: {fp['score']:.3f}, 最大IoU: {fp['max_iou_any_gt']:.2f}")

        if stats['DUP_details']:
            print("    [详细重复检测 (DUP)]: ")
            for dup in stats['DUP_details']:
                print(
                    f"      - [重复预测] 类别: {dup['pred_label']:<5} (目标: {dup['gt_match_label']}), 置信度: {dup['score']:.3f}")
    print("=" * 80)


if __name__ == '__main__':
    main()