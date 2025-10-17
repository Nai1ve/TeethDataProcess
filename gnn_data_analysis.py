import torch
from collections import defaultdict
import os

# --- 配置 ---
# 请将此路径指向你的GNN *训练* 数据集文件
GNN_TRAIN_DATA_PATH = 'gnn_data/gnn_train_data_0.05.pt'

# GNN标签中代表“背景”的类别索引
BACKGROUND_CLASS_INDEX = 48


def analyze_dataset_composition(data_path):
    """
    分析GNN数据集的构成，统计TP, FP-Class, 和背景节点的数量。
    """
    if not os.path.exists(data_path):
        print(f"错误: 数据文件未找到 at '{data_path}'")
        return

    print(f"正在加载GNN训练数据从: {data_path}...")
    try:
        data_list = torch.load(data_path)
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    if not isinstance(data_list, list) or len(data_list) == 0:
        print("错误: 文件内容不是一个有效的GNN数据列表。")
        return

    # 初始化计数器
    counters = {
        'total_nodes': 0,
        'tp_nodes': 0,
        'fp_class_nodes': 0,  # GNN需要学习纠正的节点
        'fp_background_nodes': 0  # GNN需要学习抑制的节点
    }

    print("开始分析节点构成...")
    # 遍历数据集中的每一个图 (Data对象)
    for i, data in enumerate(data_list):
        num_nodes_in_graph = data.num_nodes
        counters['total_nodes'] += num_nodes_in_graph

        # 获取GNN的真实标签 (y) 和 Faster R-CNN的原始预测标签
        gnn_ground_truth = data.y
        rcnn_original_preds = data.pred_labels_raw

        # 检查数据完整性
        if gnn_ground_truth is None or rcnn_original_preds is None or len(
                gnn_ground_truth) != num_nodes_in_graph or len(rcnn_original_preds) != num_nodes_in_graph:
            print(f"警告: 图 {i} 的数据不完整或维度不匹配，已跳过。")
            continue

        # 遍历图中的每一个节点
        for j in range(num_nodes_in_graph):
            gnn_gt_label = gnn_ground_truth[j].item()
            rcnn_pred_label = rcnn_original_preds[j].item()

            if gnn_gt_label == BACKGROUND_CLASS_INDEX:
                # 如果GNN的目标是将其分类为背景，那么这是一个需要被抑制的FP
                counters['fp_background_nodes'] += 1
            else:
                # 如果GNN的目标是将其分类为某个牙齿
                if gnn_gt_label == rcnn_pred_label:
                    # GNN目标与R-CNN原始预测一致，这是一个TP节点
                    counters['tp_nodes'] += 1
                else:
                    # GNN目标与R-CNN原始预测不一致，这是一个需要被纠正的FP-Class节点
                    counters['fp_class_nodes'] += 1

    # --- 打印最终的分析报告 ---
    print("\n" + "=" * 50)
    print("--- GNN 训练数据构成分析报告 ---")
    print("=" * 50)
    total = counters['total_nodes']
    if total == 0:
        print("数据集中没有找到任何节点。")
        return

    tp_count = counters['tp_nodes']
    fp_class_count = counters['fp_class_nodes']
    fp_bg_count = counters['fp_background_nodes']

    # 确保计数总和正确
    assert (tp_count + fp_class_count + fp_bg_count) == total, "节点计数总和不匹配!"

    print(f"总计节点数: {total}")
    print("-" * 20)
    print(f"  -> '确认'型节点 (TPs):          {tp_count:>6} ({tp_count / total:.2%})")
    print(f"  -> '纠正'型节点 (FP-Class):      {fp_class_count:>6} ({fp_class_count / total:.2%})")
    print(f"  -> '抑制'型节点 (FP-Background): {fp_bg_count:>6} ({fp_bg_count / total:.2%})")
    print("-" * 20)

    # 核心结论
    correction_ratio = fp_class_count / (fp_class_count + fp_bg_count + 1e-6)
    print("\n核心发现:")
    print(f"在所有需要模型做出改变的'错误'节点中 ({fp_class_count + fp_bg_count}个):")
    print(f"  -> 需要'纠正'的 (FP-Class) 占比: {correction_ratio:.2%}")
    print(f"  -> 需要'抑制'的 (Background) 占比: {1 - correction_ratio:.2%}")
    print("=" * 50)


if __name__ == '__main__':
    analyze_dataset_composition(GNN_TRAIN_DATA_PATH)
