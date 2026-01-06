#!/usr/bin/env python3
"""
统一配置 - 确保训练、评估、预测的一致性
"""

# 模型配置 - 极致精度版本 (目标mIoU>85%)
MODEL_CONFIG = {
    'rgb_channels': 3,
    'dsm_channels': 1,
    'num_classes': 6,
    'embed_dim': 768,  # ⭐ 384→768，匹配ViT-Base，充分发挥模块潜力
    'enable_remote_sensing_innovations': True,  # 启用遥感创新模块以提升精度
    'use_multi_scale_aggregator': False,  # 禁用MultiScaleContextAggregator (DataParallel不兼容)
    'use_simple_mode': True,  # 简化模式，提高DataParallel稳定性
    
    # Backbone配置 (简化ResNet - DataParallel稳定)
    'backbone': 'resnet',  # 使用简化ResNet (ConvNeXt在DataParallel下不稳定)
    
    # 预训练权重配置 - ⭐ 启用预训练以加速收敛
    'pretrained': True,  # ⭐ False→True，使用ImageNet21k预训练权重
    'pretrained_model': 'R50-ViT-B_16',  # 使用ResNet50+ViT-B/16预训练模型
    'pretrained_path': None,  # 将在下方根据环境动态设置
    'freeze_backbone': False,  # 不冻结backbone，允许微调
    
    # 可用的预训练模型选项 (在pretrained_weights目录中):
    # - 'R50-ViT-B_16': ResNet50 + ViT-Base/16 (推荐, embed_dim=768)
    # - 'ViT-B_16': ViT-Base/16 (embed_dim=768)
    # - 'ViT-B_32': ViT-Base/32 (embed_dim=768)
    # - 'ViT-L_16': ViT-Large/16 (embed_dim=1024, 需要更改embed_dim)
    # - 'ViT-L_32': ViT-Large/32 (embed_dim=1024, 需要更改embed_dim)
    # - 'ViT-H_14': ViT-Huge/14 (embed_dim=1280, 需要更改embed_dim)
    # 注意：使用预训练需要embed_dim=768，当前为384以节省内存
    
    # 注意: ConvNeXt需要DDP模式，DataParallel不兼容
}

# 数据配置 - 极致精度版本
DATA_CONFIG = {
    'window_size': (512, 512),  # ⭐ 256→512，增大上下文窗口，提升大目标识别
    'train_stride': 128,  # ⭐ 64→128，训练时步长（平衡速度和覆盖）
    'eval_stride': 256,  # ⭐ 128→256，训练时验证步长（加速验证）
    
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
        'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE',  # 优先级1: complete标签（完整标签，无边界擦除）⭐
        'gts_eroded_for_participants',  # 优先级2: eroded标签（FTransUNet标准）
        'gts_for_participants'  # 优先级3: 标准标签
    ],
    'label_suffix': '.tif',  # complete标签的后缀 ⭐
    'label_conversion': 'convert_from_color',  # 使用FTransUNet的convert_from_color方法
}

# 训练配置 - 极致精度版本 (目标mIoU>85%)
TRAIN_CONFIG = {
    'epochs': 300,  # ⭐ 200→300，更长训练以充分收敛
    'batch_size': 2,  # ⭐ 4→2，512窗口需要更多内存，减小batch_size
    'num_workers': 4,  # 数据加载线程数
    
    # 优化器配置 - 稳定训练设置
    'optimizer': 'AdamW',  # 使用AdamW（比SGD更稳定，收敛更快）
    'initial_lr': 0.00005,  # ⭐ 降低学习率以提高训练稳定性
    'momentum': 0.9,  # SGD momentum（AdamW不使用）
    'weight_decay': 0.01,  # ⭐ 适度正则化，避免过度惩罚
    'max_grad_norm': 1.0,  # 梯度裁剪
    
    # 学习率调度器 - Cosine退火最优
    'scheduler': 'CosineAnnealingLR',  # 使用Cosine退火
    'scheduler_milestones': [150, 225, 270],  # 备用milestones
    'scheduler_gamma': 0.1,  # 备用gamma
    'cosine_t_max': 300,  # ⭐ 200→300，Cosine周期（匹配epochs）
    'cosine_eta_min': 0.000001,  # ⭐ 0.00001→0.000001，更小学习率精细调整
    
    # 学习率预热 - 稳定训练设置
    'use_warmup': True,  # 启用学习率预热
    'warmup_epochs': 20,  # ⭐ 预训练模型需要更长预热
    'warmup_start_lr': 0.000005,  # ⭐ 更小的起始学习率以提高稳定性
    
    # 损失函数配置 - 平衡类别权重
    'class_weights': [1.0, 2.0, 3.0, 2.0, 8.0, 10.0],  # ⭐ 适度提高稀有类别权重
    # 0:Impervious(1.0-最常见), 1:Building(2.0), 2:LowVeg(3.0), 3:Tree(2.0), 4:Car(8.0-稀少), 5:Clutter(10.0-最稀少)
    'use_simple_loss': False,  # 使用完整混合损失
    'focal_loss_gamma': 2.0,  # ⭐ 适度的难样本聚焦
    'dice_loss_weight': 0.4,  # Dice Loss权重
    'aux_loss_weight': 0.3,  # 辅助损失权重
    'loss_weights': {
        'ce': 0.20,  # ⭐ 0.25→0.20，CrossEntropy权重
        'focal': 0.25,  # ⭐ 0.15→0.25，Focal Loss权重（更强处理难分类样本）
        'dice': 0.30,  # ⭐ 0.35→0.30，Dice Loss权重
        'iou': 0.20,  # IoU Loss权重（直接优化目标指标）
        'boundary': 0.05,  # Boundary Loss权重（边界精细化）
    },  # 总权重 = 1.0
    
    # 训练策略 - 极致强化少数类别采样
    'use_class_biased_sampling': True,  # 使用类别偏置采样
    'class_0_sampling_prob': 0.05,  # ⭐ 0.10→0.05，类0采样概率（大幅降低）
    'class_1_sampling_prob': 0.15,  # ⭐ 0.20→0.15，类1采样概率
    'class_2_sampling_prob': 0.20,  # 类2采样概率
    'class_3_sampling_prob': 0.15,  # 类3采样概率
    'class_4_sampling_prob': 0.25,  # ⭐ 0.20→0.25，类4采样概率（Car类别，大幅提高）
    'class_5_sampling_prob': 0.30,  # ⭐ 0.25→0.30，类5采样概率（Clutter类别，最高）
    
    # 数据增强 - 强增强提升泛化
    'use_strong_augmentation': True,  # ⭐ False→True，启用强数据增强
    'augmentation_prob': 0.8,  # ⭐ 0.6→0.8，数据增强概率（提高到80%）
    'augmentations': {  # 基础增强策略
        'random_rotation': (-10, 10),  # 小角度旋转
        'random_brightness_contrast': (0.8, 1.2, 0.8, 1.2),  # 模拟光照变化
        'random_flip': 0.5,  # 水平/垂直翻转
    },
    'use_label_smoothing': False,  # 关闭标签平滑
    'label_smoothing': 0.0,  # 标签平滑系数
    
    # 验证策略
    'val_frequency': 1,  # 每个epoch验证一次
    'save_best_only': True,  # 只保存最佳模型
    'patience': 50,  # ⭐ 30→50，早停耐心值（增加到50，给模型更多时间）
}

# 评估配置（优化：极致精度，预测图与GT高度一致）
EVAL_CONFIG = {
    'stride': 64,  # ⭐ 4→64，最终评估步长（平衡精度和速度）
    'window_size': 512,  # ⭐ 256→512，评估窗口大小（匹配训练）
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
    'stride': 64,  # ⭐ 16→64，预测步长（与评估一致）
    'window_size': 512,  # ⭐ 256→512，预测窗口大小（匹配训练）
    'accumulation_method': 'logits',  # 累积方法：'logits' 或 'probabilities'
    'postprocess': EVAL_CONFIG['postprocess'],  # 使用与评估相同的后处理配置
}

# HPC路径配置
HPC_CONFIG = {
    'base_path': '/scratch/lixuyang/ACF-Network-Full',
    'data_path': '/scratch/lixuyang/ACF-Network-Full/data',
    'pretrained_weights_path': '/scratch/lixuyang/ACF-Network-Full/pretrained_weights',
    'eroded_labels_path': '/scratch/lixuyang/ACF-Network-Full/data/Vaihingen/gts_eroded_for_participants',
    'complete_labels_path': '/scratch/lixuyang/ACF-Network-Full/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE',
}

# 本地路径配置
LOCAL_CONFIG = {
    'base_path': './',
    'data_path': './data',
    'pretrained_weights_path': './pretrained_weights',
    'eroded_labels_path': './data/Vaihingen/gts_eroded_for_participants',
    'complete_labels_path': './data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE',
}

# 根据环境自动选择路径配置
import os
if os.path.exists(HPC_CONFIG['base_path']):
    PATH_CONFIG = HPC_CONFIG
else:
    PATH_CONFIG = LOCAL_CONFIG

# 动态设置预训练权重路径
MODEL_CONFIG['pretrained_path'] = os.path.join(
    PATH_CONFIG['pretrained_weights_path'],
    'imagenet21k+imagenet2012',
    'imagenet21k+imagenet2012_R50+ViT-B_16.npz'
)

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

