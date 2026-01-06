#!/usr/bin/env python3
"""
统一配置 - 确保训练、评估、预测的一致性
"""

# 模型配置
MODEL_CONFIG = {
    'rgb_channels': 3,
    'dsm_channels': 1,
    'num_classes': 6,
    'embed_dim': 384,  # 提升到384以达到mIoU>85%的目标 (更强的特征表达能力)
    'enable_remote_sensing_innovations': True,  # 启用遥感创新模块以提升精度
    'use_multi_scale_aggregator': False,  # 禁用MultiScaleContextAggregator (DataParallel不兼容)
    'use_simple_mode': True,  # 简化模式，提高DataParallel稳定性
    
    # Backbone配置 (简化ResNet - DataParallel稳定)
    'backbone': 'resnet',  # 使用简化ResNet (ConvNeXt在DataParallel下不稳定)
    'pretrained':  False,  # 从头训练
    # 注意: ConvNeXt需要DDP模式，DataParallel不兼容
}

# 数据配置
DATA_CONFIG = {
    'window_size': (256, 256),  # 训练和评估统一窗口大小
    'train_stride': 64,  # 训练时步长（快速训练：64）⭐
    'eval_stride': 128,  # 训练时验证步长（快速验证：128）
    
    # DSM预处理：统一使用每张图像的min-max归一化
    'dsm_normalization': 'global_min_max',  # 改为全局归一化，保留DSM高程全局一致性
    
    # 数据增强配置（最小化，最大速度）
    'use_augmentation': True,  # 启用数据增强
    'aug_flip_prob': 0.5,  # 水平翻转概率（保留，开销小）
    'aug_rotate_prob': 0.0,  # 旋转概率（禁用，加速）
    'aug_rotate_angles': [0, 90, 180, 270],  # 旋转角度
    'aug_color_jitter': False,  # 颜色抖动（禁用）
    'aug_brightness': 0.0,  # 亮度变化（禁用）
    'aug_contrast': 0.0,  # 对比度变化（禁用）
    'aug_saturation': 0.0,  # 饱和度变化（禁用）
    'dsm_global_stats': {'min': -5.0, 'max': 50.0}, # 新增：Vaihingen数据集DSM实测范围（-5~50米） # 'per_image_min_max' 或 'global_min_max'
    
    # RGB预处理：统一使用/255.0归一化
    'rgb_normalization': 'divide_255',
    
    # 标签配置
    'label_path_priority': [
        'gts_eroded_for_participants',  # 优先级1: eroded标签（FTransUNet标准）
        'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE',  # 优先级2: complete标签
        'gts_for_participants'  # 优先级3: 标准标签
    ],
    'label_suffix': '_noBoundary.tif',  # eroded标签的后缀
    'label_conversion': 'convert_from_color',  # 使用FTransUNet的convert_from_color方法
}

# 训练配置 - 极致优化版本（目标：OA>92%, mIoU>86%，预测图与GT高度一致）
TRAIN_CONFIG = {
    'epochs': 180,  # 训练轮数（平衡方案：180轮）⭐
    'batch_size': 12,  # 批大小（4卡GPU，每卡3，小batch更快更准）⭐
    'num_workers': 4,  # 数据加载线程数
    
    # 优化器配置（优化版：提高学习率，加快收敛）
    'optimizer': 'SGD',  # 使用SGD（对比算法都用SGD）
    'initial_lr': 0.02,  # 初始学习率（提高到0.02）⭐
    'momentum': 0.9,  # SGD momentum
    'weight_decay': 0.0001,  # 权重衰减
    'max_grad_norm': 1.0,  # 梯度裁剪
    
    # 学习率调度器（优化版：Cosine退火，更平滑）⭐
    'scheduler': 'CosineAnnealingLR',  # 使用Cosine退火
    'scheduler_milestones': [100, 200, 250],  # 备用milestones
    'scheduler_gamma': 0.1,  # 备用gamma
    'cosine_t_max': 180,  # Cosine周期
    'cosine_eta_min': 0.0001,  # 最小学习率
    
    # 学习率预热（延长预热时间）
    'use_warmup': True,  # 启用学习率预热
    'warmup_epochs': 10,  # 预热轮数（延长到10个epoch）⭐
    'warmup_start_lr': 0.002,  # 预热起始学习率
    
    # 损失函数配置（优化版：平衡类别权重）⭐
    'class_weights': [1.0, 1.0, 2.5, 1.5, 6.0, 8.0],  # 类别权重（降低Clutter权重避免训练不稳定）
    # 0:Impervious(1.0), 1:Building(1.0), 2:LowVeg(2.5), 3:Tree(1.5), 4:Car(6.0), 5:Clutter(8.0)
    'use_simple_loss': False,  # 使用简化混合损失（CE + Dice + IoU + Boundary）
    'focal_loss_gamma': 2.0,  # Focal Loss gamma（降低到2.0）
    'dice_loss_weight': 0.4,  # Dice Loss权重
    'aux_loss_weight': 0.3,  # 辅助损失权重
    'loss_weights': {
        'ce': 0.35,  # CrossEntropy权重
        'focal': 0.0,  # Focal Loss权重（保持关闭）
        'dice': 0.45,  # Dice Loss权重（提高以优化IoU）
        'iou': 0.15,  # IoU Loss权重（提高以直接优化mIoU）
        'boundary': 0.05,  # Boundary Loss权重（降低）
    },  # 总权重 = 1.0
    
    # 训练策略（简化版：平衡采样策略）
    'use_class_biased_sampling': True,  # 使用类别偏置采样
    'class_0_sampling_prob': 0.20,  # 类0采样概率
    'class_1_sampling_prob': 0.20,  # 类1采样概率
    'class_2_sampling_prob': 0.15,  # 类2采样概率
    'class_3_sampling_prob': 0.15,  # 类3采样概率
    'class_4_sampling_prob': 0.18,  # 类4采样概率（Car类别）
    'class_5_sampling_prob': 0.22,  # 类5采样概率（Clutter类别，提高采样）
    
    # 数据增强（简化版：关闭强增强和标签平滑）
    'use_strong_augmentation': False,  # 关闭强数据增强
    'augmentation_prob': 0.5,  # 数据增强概率（提高以增强泛化能力）
    'augmentations': {  # 基础增强策略
        'random_rotation': (-10, 10),  # 小角度旋转
        'random_brightness_contrast': (0.8, 1.2, 0.8, 1.2),  # 模拟光照变化
        'random_flip': 0.5,  # 水平/垂直翻转
    },
    'use_label_smoothing': False,  # 关闭标签平滑（让模型直接学习）
    'label_smoothing': 0.0,  # 标签平滑系数
    
    # 验证策略
    'val_frequency': 1,  # 每个epoch验证一次
    'save_best_only': True,  # 只保存最佳模型
    'patience': 20,  # 早停耐心值（增加耐心值，允许更多训练）
}

# 评估配置（优化：极致精度，预测图与GT高度一致）
EVAL_CONFIG = {
    'stride': 4,  # 最终评估步长（极致精度，目标OA>90%）
    'window_size': 256,  # 评估窗口大小
    'postprocess': {
        'min_area': 50,  # 最小连通域面积（200→100，更严格移除小误分区域）
        'morphology': {
            'closing_size': 5,  # 闭运算结构元素大小（5→7，更强填充小洞）
            'opening_size': 5,  # 开运算结构元素大小（3→5，更强去除小突起）
        },
        'confidence_threshold': 0.55,  # 置信度阈值（0.5→0.55，更严格修正低置信度区域）
        'use_crf_smoothing': True,  # 启用CRF-like平滑（增强空间一致性）
        'crf_sigma': 1.5,  # CRF平滑sigma
    },
    'metrics': {
        'compute_class_wise': True,  # 计算各类别指标
        'compute_confusion_matrix': True,  # 计算混淆矩阵
        'use_top5_miou': True,  # 使用前5类mIoU（Vaihingen标准）
    }
}

# 预测配置
PREDICT_CONFIG = {
    'stride': 16,  # 预测步长（与评估一致）
    'window_size': 256,  # 预测窗口大小
    'accumulation_method': 'logits',  # 累积方法：'logits' 或 'probabilities'
    'postprocess': EVAL_CONFIG['postprocess'],  # 使用与评估相同的后处理配置
}

# HPC路径配置
HPC_CONFIG = {
    'base_path': '/project/lixuyang/collaborative_framework_project',
    'data_path': '/project/lixuyang/collaborative_framework_project/data',
    'eroded_labels_path': '/project/lixuyang/collaborative_framework_project/data/Vaihingen/gts_eroded_for_participants',
    'complete_labels_path': '/project/lixuyang/collaborative_framework_project/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE',
}

# 本地路径配置
LOCAL_CONFIG = {
    'base_path': './',
    'data_path': './data',
    'eroded_labels_path': './data/Vaihingen/gts_eroded_for_participants',
    'complete_labels_path': './data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE',
}

# 根据环境自动选择路径配置
import os
if os.path.exists(HPC_CONFIG['base_path']):
    PATH_CONFIG = HPC_CONFIG
else:
    PATH_CONFIG = LOCAL_CONFIG

# 统一配置导出
UNIFIED_CONFIG = {
    'model': MODEL_CONFIG,
    'data': DATA_CONFIG,
    'train': TRAIN_CONFIG,
    'eval': EVAL_CONFIG,
    'predict': PREDICT_CONFIG,
    'paths': PATH_CONFIG
}

if __name__ == '__main__':
    import json
    print("统一配置:")
    print(json.dumps(UNIFIED_CONFIG, indent=2, default=str))

