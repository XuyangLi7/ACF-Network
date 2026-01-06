# Collaborative Framework Project 666 - 关键配置分析报告

## 📁 已复制的关键文件

1. **unified_config.py** - 统一配置文件
2. **universal_dataset.py** - 数据集加载器
3. **train_enhanced.py** - 训练脚本
4. **enhanced_multimodal_framework.py** - 模型框架

---

## 🏷️ 标签配置

### 使用的标签类型：**COMPLETE 标签（完整6类）**

**配置位置**：`unified_config.py` → `DATA_CONFIG` → `label_path_priority`

```python
'label_path_priority': [
    'gts_eroded_for_participants',  # 优先级1: eroded标签（FTransUNet标准）
    'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE',  # 优先级2: complete标签
    'gts_for_participants'  # 优先级3: 标准标签
]
```

**实际使用**：`universal_dataset.py` 中明确指定：
```python
'label': 'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/top_mosaic_09cm_area{}.tif'
```

**标签后缀**：`_noBoundary.tif`（eroded标签的后缀）

**标签转换方法**：`convert_from_color` - 使用FTransUNet的RGB颜色编码转换

### 标签颜色映射（6类 + 1个未定义类）

```python
palette = {
    (255, 255, 255): 0,  # Impervious surfaces (白色) - 不透水表面
    (0, 0, 255): 1,      # Buildings (蓝色) - 建筑物
    (0, 255, 255): 2,    # Low vegetation (青色) - 低植被
    (0, 255, 0): 3,      # Trees (绿色) - 树木
    (255, 255, 0): 4,    # Cars (黄色) - 汽车
    (255, 0, 0): 5,      # Clutter (红色) - 杂物
    (0, 0, 0): 6         # Undefined (黑色) - 未定义
}
```

---

## 💾 数据预处理方式

### 1. RGB预处理

**归一化方法**：`divide_255`
```python
'rgb_normalization': 'divide_255'
```

**实际操作**：
```python
rgb_data = rgb_data.astype(np.float32) / 255.0  # 归一化到[0, 1]
```

**通道顺序**：取前3个通道 (NIR, R, G)
```python
rgb_data = rgb[:, :, :3].transpose((2, 0, 1))  # (H, W, 3) → (3, H, W)
```

### 2. DSM预处理

**归一化方法**：`global_min_max`（全局最小-最大归一化）
```python
'dsm_normalization': 'global_min_max'
```

**全局统计量**：
```python
'dsm_global_stats': {'min': -5.0, 'max': 50.0}  # Vaihingen数据集DSM实测范围（-5~50米）
```

**实际操作**：
```python
# 1. 裁剪到范围
dsm = np.clip(dsm, min_val, max_val)  # min=-5.0, max=50.0

# 2. 归一化到[0, 1]
dsm = (dsm - min_val) / (max_val - min_val)

# 3. 添加通道维度
dsm = dsm[np.newaxis, :, :]  # (H, W) → (1, H, W)
```

### 3. 数据增强

**基础增强**（训练集）：
```python
'use_augmentation': True
'aug_flip_prob': 0.5  # 水平翻转概率
'aug_rotate_prob': 0.0  # 旋转概率（禁用）
'aug_color_jitter': False  # 颜色抖动（禁用）
```

**增强概率**：
```python
'augmentation_prob': 0.5  # 50%概率进行数据增强
```

**增强操作**：
- ✅ 垂直翻转（50%概率）
- ✅ 水平镜像（50%概率）
- ❌ 旋转（禁用，加速训练）
- ❌ 颜色抖动（禁用）

**强增强**：
```python
'use_strong_augmentation': False  # 关闭强数据增强
```

### 4. 窗口滑动

**窗口大小**：
```python
'window_size': (256, 256)  # 训练和评估统一窗口大小
```

**训练步长**：
```python
'train_stride': 64  # 训练时步长（快速训练）
```

**验证步长**：
```python
'eval_stride': 128  # 训练时验证步长（快速验证）
```

**最终评估步长**：
```python
EVAL_CONFIG['stride']: 4  # 最终评估步长（极致精度）
```

---

## 🎯 损失函数配置

### 损失函数组合

**使用的损失函数**：
```python
'use_simple_loss': False  # 使用完整混合损失
```

**损失函数权重**：
```python
'loss_weights': {
    'ce': 0.35,       # CrossEntropy权重
    'focal': 0.0,     # Focal Loss权重（关闭）
    'dice': 0.45,     # Dice Loss权重（提高以优化IoU）
    'iou': 0.15,      # IoU Loss权重（直接优化mIoU）
    'boundary': 0.05  # Boundary Loss权重（降低）
}
# 总权重 = 1.0
```

### 类别权重

**类别平衡策略**：
```python
'class_weights': [1.0, 1.0, 2.5, 1.5, 6.0, 8.0]
```

**详细说明**：
- 类别0 (Impervious): 1.0 - 最常见，权重最低
- 类别1 (Building): 1.0 - 常见
- 类别2 (Low vegetation): 2.5 - 中等权重
- 类别3 (Tree): 1.5 - 中等权重
- 类别4 (Car): 6.0 - 少数类，高权重
- 类别5 (Clutter): 8.0 - 最少数类，最高权重

### Focal Loss配置

```python
'focal_loss_gamma': 2.0  # Focal Loss gamma参数
```

### 辅助损失

```python
'aux_loss_weight': 0.3  # 辅助损失权重
```

### 标签平滑

```python
'use_label_smoothing': False  # 关闭标签平滑
'label_smoothing': 0.0
```

---

## 🎓 训练配置

### 基础参数

```python
'epochs': 180  # 训练轮数
'batch_size': 12  # 批大小（4卡GPU，每卡3）
'num_workers': 4  # 数据加载线程数
```

### 优化器

```python
'optimizer': 'SGD'  # 使用SGD
'initial_lr': 0.02  # 初始学习率
'momentum': 0.9  # SGD momentum
'weight_decay': 0.0001  # 权重衰减
'max_grad_norm': 1.0  # 梯度裁剪
```

### 学习率调度

**调度器类型**：
```python
'scheduler': 'CosineAnnealingLR'  # 使用Cosine退火
```

**Cosine参数**：
```python
'cosine_t_max': 180  # Cosine周期
'cosine_eta_min': 0.0001  # 最小学习率
```

### 学习率预热

```python
'use_warmup': True  # 启用学习率预热
'warmup_epochs': 10  # 预热轮数
'warmup_start_lr': 0.002  # 预热起始学习率
```

### 类别偏置采样

**采样策略**：
```python
'use_class_biased_sampling': True  # 使用类别偏置采样
```

**各类别采样概率**：
```python
'class_0_sampling_prob': 0.20  # Impervious
'class_1_sampling_prob': 0.20  # Building
'class_2_sampling_prob': 0.15  # Low vegetation
'class_3_sampling_prob': 0.15  # Tree
'class_4_sampling_prob': 0.18  # Car
'class_5_sampling_prob': 0.22  # Clutter（最高优先级）
```

**采样优先级**（从高到低）：
1. 类别5 (Clutter) - 70%概率
2. 类别4 (Car) - 40%概率
3. 类别2 (Low vegetation) - 30%概率
4. 类别1 (Building) - 25%概率
5. 类别3 (Tree) - 15%概率
6. 类别0 (Impervious) - 10%概率

---

## 📊 模型配置

### 基础参数

```python
'rgb_channels': 3
'dsm_channels': 1
'num_classes': 6
'embed_dim': 384  # 提升到384以达到mIoU>85%的目标
```

### 创新模块

```python
'enable_remote_sensing_innovations': True  # 启用遥感创新模块
'use_multi_scale_aggregator': False  # 禁用（DataParallel不兼容）
'use_simple_mode': True  # 简化模式，提高DataParallel稳定性
```

### Backbone

```python
'backbone': 'resnet'  # 使用简化ResNet
'pretrained': False  # 从头训练
```

---

## 🔍 验证与评估配置

### 验证策略

```python
'val_frequency': 1  # 每个epoch验证一次
'save_best_only': True  # 只保存最佳模型
'patience': 20  # 早停耐心值
```

### 评估指标

```python
'metrics': {
    'compute_class_wise': True,  # 计算各类别指标
    'compute_confusion_matrix': True,  # 计算混淆矩阵
    'use_top5_miou': True  # 使用前5类mIoU（Vaihingen标准）
}
```

### 后处理

```python
'postprocess': {
    'min_area': 50,  # 最小连通域面积
    'morphology': {
        'closing_size': 5,  # 闭运算结构元素大小
        'opening_size': 5   # 开运算结构元素大小
    },
    'confidence_threshold': 0.55,  # 置信度阈值
    'use_crf_smoothing': True,  # 启用CRF-like平滑
    'crf_sigma': 1.5
}
```

---

## 📈 训练数据集划分

### Vaihingen数据集

**训练集ID**：
```python
'train_ids': ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
# 共12张图像
```

**测试集ID**：
```python
'test_ids': ['5', '21', '15', '30']
# 共4张图像
```

---

## 🎯 关键创新点总结

### 1. 标签策略
- ✅ 使用**COMPLETE标签**（完整6类）
- ✅ FTransUNet标准的颜色编码转换
- ✅ 支持eroded标签后缀

### 2. 数据预处理
- ✅ RGB: `/255.0`归一化
- ✅ DSM: 全局min-max归一化（-5~50米）
- ✅ 类别偏置采样（优先采样少数类）

### 3. 损失函数
- ✅ 混合损失：CE(35%) + Dice(45%) + IoU(15%) + Boundary(5%)
- ✅ 类别权重平衡（Clutter:8.0, Car:6.0）
- ✅ 辅助损失（30%权重）

### 4. 训练策略
- ✅ SGD优化器（lr=0.02）
- ✅ Cosine退火学习率调度
- ✅ 10 epoch预热
- ✅ 类别偏置采样（Clutter优先级最高）

### 5. 数据增强
- ✅ 基础增强：翻转+镜像（50%概率）
- ❌ 强增强：关闭（加速训练）
- ❌ 颜色抖动：关闭

---

## 📝 使用建议

1. **标签选择**：项目使用COMPLETE标签，确保数据路径正确
2. **DSM范围**：Vaihingen数据集DSM范围为-5~50米，其他数据集需调整
3. **类别平衡**：Clutter和Car类别权重最高，训练时会优先采样
4. **学习率**：初始lr=0.02，使用Cosine退火，10 epoch预热
5. **批大小**：batch_size=12（4卡×3），可根据GPU内存调整
6. **验证频率**：每个epoch验证一次，保存最佳模型

---

## 🚀 快速启动命令

```bash
# 训练
python train_enhanced.py \
    --data_path ./data \
    --dataset_name vaihingen \
    --output_dir ./checkpoints \
    --epochs 180 \
    --batch_size 12 \
    --amp  # 使用混合精度训练

# 恢复训练
python train_enhanced.py \
    --resume ./checkpoints/last_model.pth \
    --amp
```

---

生成时间：2026-01-05
项目：collaborative_framework_project666
