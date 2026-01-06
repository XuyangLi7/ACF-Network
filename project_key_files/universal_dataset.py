#!/usr/bin/env python3
"""
统一多模态数据集加载器
支持任意模态和数据集
"""

import torch
import torch.utils.data as data
import numpy as np
import os
import logging
import random
from skimage import io
from sklearn.metrics import confusion_matrix
import itertools
from torchvision.utils import make_grid
from PIL import Image

logger = logging.getLogger(__name__)

class UniversalMultiModalDataset(data.Dataset):
    """统一多模态数据集"""
    
    def __init__(self, data_path: str, dataset_name: str = 'vaihingen', split: str = 'train',
                 window_size: tuple = (256, 256), stride: int = 32, cache: bool = True):
        super().__init__()
        
        self.data_path = data_path
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.cache = cache
        
        # 获取数据集配置
        self.config = self._get_dataset_config()
        
        # 获取文件路径
        self.data_files = self._get_file_paths()
        
        # 检查文件
        self._check_files()
        
        # 初始化缓存
        self.data_cache_ = {}
        self.label_cache_ = {}
        
        logger.info(f"Universal Multi-Modal {self.dataset_name.upper()} {split} 数据集初始化完成")
        logger.info(f"数据文件: {len(self.data_files)} 个")
        logger.info(f"窗口大小: {self.window_size}, 步长: {self.stride}")
    
    def _get_dataset_config(self):
        """获取数据集配置"""
        configs = {
            'vaihingen': {
                'modalities': ['rgb', 'dsm'],
                'train_ids': ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37'],
                'test_ids': ['5', '21', '15', '30'],
                'n_classes': 6,
                'class_names': ['道路', '建筑', '低植被', '树木', '汽车', '杂物'],
                'file_patterns': {
                    'rgb': 'top/top_mosaic_09cm_area{}.tif',
                    'dsm': 'DSM/dsm_09cm_matching_area{}.tif',
                    # 优先使用完整6类标签（本地相对路径）；训练与验证强制对齐完整标签
                    'label': 'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/top_mosaic_09cm_area{}.tif'
                }
            },
            'houston': {
                'modalities': ['hsi', 'dsm'],
                'train_ids': ['train'],
                'test_ids': ['test'],
                'n_classes': 15,
                'class_names': ['健康草', '枯草', '土壤', '水', '住宅', '商业', '道路', '高速公路', '铁路', '停车场1', '停车场2', '网球场', '跑道', '树木', '阴影'],
                'file_patterns': {
                    'hsi': 'data_HS_HR.mat',
                    'dsm': 'data_DSM_HR.mat',
                    'label': 'gt.mat'
                }
            },
            'muufl': {
                'modalities': ['hsi', 'lidar'],
                'train_ids': ['train'],
                'test_ids': ['test'],
                'n_classes': 11,
                'class_names': ['树木', '大部分草地', '混合地面', '泥土/沙子', '道路', '水', '建筑物阴影', '建筑物', '人行道', '黄色警戒线', '停车场'],
                'file_patterns': {
                    'hsi': 'HSI.mat',
                    'lidar': 'LiDAR2.mat',
                    'label': 'tr_ts_gt_150samples.mat'
                }
            },
            'trento': {
                'modalities': ['hsi', 'lidar'],
                'train_ids': ['train'],
                'test_ids': ['test'],
                'n_classes': 6,
                'class_names': ['苹果', '建筑', '地面', '葡萄', '森林', '阴影'],
                'file_patterns': {
                    'hsi': 'HSI.mat',
                    'lidar': 'LiDAR.mat',
                    'label': 'gt.mat'
                }
            },
            'augsburg': {
                'modalities': ['hsi', 'dsm'],
                'train_ids': ['train'],
                'test_ids': ['test'],
                'n_classes': 7,
                'class_names': ['树木', '低密度住宅', '高密度住宅', '土壤', '街道', '铁路', '商业'],
                'file_patterns': {
                    'hsi': 'data_HS_LR.mat',
                    'dsm': 'data_DSM.mat',
                    'label': 'train_test_gt.mat'
                }
            },
            'berlin': {
                'modalities': ['hsi', 'sar'],
                'train_ids': ['train'],
                'test_ids': ['test'],
                'n_classes': 8,
                'class_names': ['森林', '住宅区', '工业区', '低植被', '土壤', '水', '铁路', '商业区'],
                'file_patterns': {
                    'hsi': 'data_HS_LR.mat',
                    'sar': 'data_SAR_HR.mat',
                    'label': 'TrainImage.mat'
                }
            }
        }
        
        return configs.get(self.dataset_name, configs['vaihingen'])
    
    def _get_file_paths(self):
        """获取文件路径"""
        config = self.config
        
        if self.dataset_name == 'vaihingen':
            # Vaihingen数据集特殊处理
            ids = config['train_ids'] if self.split == 'train' else config['test_ids']
            data_files = []
            
            for id in ids:
                file_paths = {}
                for modality, pattern in config['file_patterns'].items():
                    file_path = os.path.join(self.data_path, 'Vaihingen', pattern.format(id))
                    file_paths[modality] = file_path
                data_files.append(file_paths)
            
            return data_files
        else:
            # 其他数据集处理
            data_files = []
            for modality, filename in config['file_patterns'].items():
                file_path = os.path.join(self.data_path, self.dataset_name, filename)
                if not hasattr(self, 'file_paths'):
                    self.file_paths = {}
                self.file_paths[modality] = file_path
            
            return [self.file_paths]
    
    def _check_files(self):
        """检查文件是否存在"""
        for file_dict in self.data_files:
            for modality, file_path in file_dict.items():
                if not os.path.isfile(file_path):
                    logger.warning(f"文件不存在: {file_path}")
    
    def __len__(self):
        """返回数据集大小"""
        if not hasattr(self, '_total_windows'):
            self._calculate_total_windows()
        return self._total_windows
    
    def _calculate_total_windows(self):
        """计算总的滑动窗口数量"""
        total_windows = 0
        
        for i, file_dict in enumerate(self.data_files):
            try:
                # 加载一个图像来获取尺寸
                if self.dataset_name == 'vaihingen':
                    rgb_file = file_dict['rgb']
                    rgb = self._load_image(rgb_file)
                    if rgb is not None:
                        H, W = rgb.shape[:2]
                        
                        # 计算滑动窗口数量
                        windows_h = (H - self.window_size[0]) // self.stride + 1
                        windows_w = (W - self.window_size[1]) // self.stride + 1
                        windows_per_image = windows_h * windows_w
                        
                        total_windows += windows_per_image
                        logger.info(f"图像 {i+1}: {H}x{W}, 滑动窗口: {windows_per_image}")
                else:
                    # 其他数据集使用固定数量
                    total_windows += 1000
                    
            except Exception as e:
                logger.warning(f"计算图像 {i+1} 窗口数量失败: {e}")
                total_windows += 1000
        
        self._total_windows = total_windows
        logger.info(f"总滑动窗口数量: {self._total_windows}")
    
    def _load_image(self, file_path):
        """加载图像"""
        try:
            if file_path.endswith('.mat'):
                # 加载MAT文件
                from scipy.io import loadmat
                mat = loadmat(file_path)
                # 获取第一个非元数据的数组
                for key in mat.keys():
                    if not key.startswith('__'):
                        return mat[key]
                return None
            else:
                # 加载图像文件
                return io.imread(file_path)
        except Exception as e:
            logger.warning(f"加载图像失败: {e}")
            return None
    
    def _load_mat_data(self, file_path, key=None):
        """加载MAT文件数据"""
        try:
            from scipy.io import loadmat
            mat = loadmat(file_path)
            if key:
                return mat[key]
            else:
                # 获取第一个非元数据的数组
                for k in mat.keys():
                    if not k.startswith('__'):
                        return mat[k]
                return None
        except Exception as e:
            logger.warning(f"加载MAT数据失败: {e}")
            return None
    
    def __getitem__(self, i):
        """获取数据项"""
        if self.dataset_name == 'vaihingen':
            return self._get_vaihingen_item(i)
        else:
            return self._get_mat_item(i)
    
    def _get_vaihingen_item(self, i):
        """获取Vaihingen数据项"""
        # 随机选择一个图像（训练阶段对类5/类4进行偏置采样）
        random_idx = random.randint(0, len(self.data_files) - 1)
        file_dict = self.data_files[random_idx]
        
        # 加载RGB数据
        if random_idx in self.data_cache_.keys():
            rgb_data = self.data_cache_[random_idx]
        else:
            try:
                rgb = self._load_image(file_dict['rgb'])
                if rgb is not None and rgb.ndim == 3 and rgb.shape[2] >= 3:
                    # 取前3个通道 (NIR, R, G)
                    rgb_data = rgb[:, :, :3].transpose((2, 0, 1))
                else:
                    rgb_data = np.random.rand(3, 1000, 1000).astype(np.float32)
                rgb_data = rgb_data.astype(np.float32) / 255.0
                
                if self.cache:
                    self.data_cache_[random_idx] = rgb_data
            except Exception as e:
                logger.warning(f"加载RGB数据失败: {e}")
                rgb_data = np.random.rand(3, 1000, 1000).astype(np.float32)
        
        # 加载DSM数据
        try:
            dsm = self._load_image(file_dict['dsm'])
            if dsm is not None:
                if dsm.ndim == 3:
                    dsm = dsm[:, :, 0]
                dsm = dsm.astype(np.float32)
                # 使用全局归一化（与配置一致）
                from unified_config import DATA_CONFIG
                dsm_stats = DATA_CONFIG.get('dsm_global_stats', {'min': -5.0, 'max': 50.0})
                min_val, max_val = dsm_stats['min'], dsm_stats['max']
                dsm = np.clip(dsm, min_val, max_val)
                dsm = (dsm - min_val) / (max_val - min_val)
                dsm = dsm[np.newaxis, :, :]  # 添加通道维度
            else:
                dsm = np.random.rand(1, 1000, 1000).astype(np.float32)
        except Exception as e:
            logger.warning(f"加载DSM数据失败: {e}")
            dsm = np.random.rand(1, 1000, 1000).astype(np.float32)
        
        # 加载标签数据
        if random_idx in self.label_cache_.keys():
            label_data = self.label_cache_[random_idx]
        else:
            try:
                label = self._load_image(file_dict['label'])
                if label is not None:
                    label_data = self.convert_from_color(label)
                else:
                    label_data = np.random.randint(0, 6, (1000, 1000))
                
                if self.cache:
                    self.label_cache_[random_idx] = label_data
                    # 预存所有类别的像素位置用于偏置采样（确保所有类别都被采样）
                    if not hasattr(self, 'class_positions'):
                        self.class_positions = {}
                    if random_idx not in self.class_positions:
                        positions = {}
                        for cls_id in range(6):  # 所有6个类别
                            ys, xs = np.where(label_data == cls_id)
                            positions[cls_id] = np.stack([ys, xs], axis=1) if ys.size > 0 else np.empty((0, 2), dtype=np.int64)
                        self.class_positions[random_idx] = positions
            except Exception as e:
                logger.warning(f"加载标签数据失败: {e}")
                label_data = np.random.randint(0, 6, (1000, 1000))
        
        # 获取位置：训练集对所有类别进行偏置采样（优先采样少数类别）
        if self.split == 'train' and hasattr(self, 'class_positions') and random_idx in self.class_positions:
            picked = False
            # 优先级：类别5(Clutter) > 类别4(Car) > 类别2(Low vegetation) > 类别1(Building) > 其他
            # 采样概率：类别5=0.7, 类别4=0.4, 类别2=0.3, 类别1=0.25, 其他=0.1
            sampling_config = [
                (5, 0.7),  # Clutter - 最高优先级
                (4, 0.4),  # Car
                (2, 0.3),  # Low vegetation
                (1, 0.25), # Building
                (3, 0.15), # Tree (虽然常见，但也采样)
                (0, 0.1)   # Impervious surfaces (最常见，采样概率最低)
            ]
            
            for cls_id, prob in sampling_config:
                coords = self.class_positions[random_idx].get(cls_id, np.empty((0, 2)))
                if coords.shape[0] > 0 and random.random() < prob:
                    y_center, x_center = coords[np.random.randint(0, coords.shape[0])]
                    h, w = self.window_size  # h=height, w=width
                    y1 = max(0, int(y_center) - h // 2)
                    x1 = max(0, int(x_center) - w // 2)
                    y2 = y1 + h
                    x2 = x1 + w
                    # 边界修正
                    if y2 > label_data.shape[0]:
                        y1 = max(0, label_data.shape[0] - h)
                        y2 = y1 + h
                    if x2 > label_data.shape[1]:
                        x1 = max(0, label_data.shape[1] - w)
                        x2 = x1 + w
                    
                    # 确保窗口尺寸正确且不超出边界
                    if (y2 - y1 != h or x2 - x1 != w or 
                        y2 > label_data.shape[0] or x2 > label_data.shape[1] or
                        y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1):
                        continue
                    picked = True
                    break
            if not picked:
                x1, x2, y1, y2 = self.get_safe_random_pos(rgb_data, self.window_size)
        else:
            x1, x2, y1, y2 = self.get_safe_random_pos(rgb_data, self.window_size)
        
        # 提取patch - 注意坐标顺序：x是宽度，y是高度
        rgb_patch = rgb_data[:, y1:y2, x1:x2]
        dsm_patch = dsm[:, y1:y2, x1:x2]
        label_patch = label_data[y1:y2, x1:x2]
        
        # 调试信息
        if rgb_patch.shape[1] != self.window_size[0] or rgb_patch.shape[2] != self.window_size[1]:
            logger.debug(f"坐标: ({x1},{y1},{x2},{y2}), RGB形状: {rgb_patch.shape}, 期望: {self.window_size}")
            logger.debug(f"图像尺寸: RGB{rgb_data.shape}, DSM{dsm.shape}, Label{label_data.shape}")
        
        # 确保patch尺寸正确
        expected_h, expected_w = self.window_size
        if rgb_patch.shape[1] != expected_h or rgb_patch.shape[2] != expected_w:
            logger.warning(f"Patch尺寸不正确: {rgb_patch.shape[1:3]}, 期望: {self.window_size}")
            
            # 检查patch是否有效
            if rgb_patch.size == 0 or dsm_patch.size == 0 or label_patch.size == 0:
                logger.warning(f"发现空patch，使用默认patch")
                # 返回一个默认的patch
                rgb_patch = np.zeros((3, expected_h, expected_w), dtype=np.float32)
                dsm_patch = np.zeros((1, expected_h, expected_w), dtype=np.float32)
                label_patch = np.zeros((expected_h, expected_w), dtype=np.uint8)
            else:
                # 使用OpenCV进行resize，避免PIL的类型问题
                import cv2
                
                # 转换RGB格式并resize
                if rgb_patch.shape[0] == 3:
                    rgb_patch = np.transpose(rgb_patch, (1, 2, 0))
                # 确保数据类型正确
                if rgb_patch.dtype != np.uint8:
                    rgb_patch = (rgb_patch * 255).astype(np.uint8)
                # 检查尺寸是否有效
                if rgb_patch.shape[0] > 0 and rgb_patch.shape[1] > 0:
                    rgb_patch = cv2.resize(rgb_patch, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                else:
                    rgb_patch = np.zeros((expected_h, expected_w, 3), dtype=np.uint8)
                if len(rgb_patch.shape) == 3:
                    rgb_patch = np.transpose(rgb_patch, (2, 0, 1))
                # 转换回float32并归一化
                rgb_patch = rgb_patch.astype(np.float32) / 255.0
                
                # 转换DSM格式并resize
                if len(dsm_patch.shape) == 3:
                    dsm_patch = dsm_patch[0]
                # 确保DSM数据类型正确
                if dsm_patch.dtype != np.uint8:
                    dsm_patch = (dsm_patch * 255).astype(np.uint8)
                # 检查尺寸是否有效
                if dsm_patch.shape[0] > 0 and dsm_patch.shape[1] > 0:
                    dsm_patch = cv2.resize(dsm_patch, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                else:
                    dsm_patch = np.zeros((expected_h, expected_w), dtype=np.uint8)
                if len(dsm_patch.shape) == 2:
                    dsm_patch = dsm_patch[np.newaxis, :, :]
                # 转换回float32并归一化
                dsm_patch = dsm_patch.astype(np.float32) / 255.0
                
                # 转换标签格式并resize
                if label_patch.shape[0] > 0 and label_patch.shape[1] > 0:
                    label_patch = cv2.resize(label_patch.astype(np.uint8), (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
                else:
                    label_patch = np.zeros((expected_h, expected_w), dtype=np.uint8)
        
        # 数据增强（根据配置的概率）
        if self.split == 'train':
            # 从统一配置读取数据增强概率
            try:
                from unified_config import TRAIN_CONFIG
                augmentation_prob = TRAIN_CONFIG.get('augmentation_prob', 0.5)
                use_strong_augmentation = TRAIN_CONFIG.get('use_strong_augmentation', False)
            except:
                augmentation_prob = 0.5
                use_strong_augmentation = False
            
            # 根据概率决定是否进行数据增强
            if random.random() < augmentation_prob:
                rgb_patch, dsm_patch, label_patch = self.data_augmentation(
                    rgb_patch, dsm_patch, label_patch,
                    flip=True, mirror=True,
                    use_strong=use_strong_augmentation
                )
        # 验证集不使用数据增强
        
        # 数据增强后再次检查尺寸（翻转和镜像不会改变尺寸，但为了保险起见）
        if rgb_patch.shape[1:] != (256, 256):
            logger.warning(f"数据增强后尺寸不正确: {rgb_patch.shape[1:]}, 期望: (256, 256)")
            # 使用OpenCV进行resize
            import cv2
            if rgb_patch.shape[0] == 3:
                rgb_patch = np.transpose(rgb_patch, (1, 2, 0))
            # 确保数据类型正确
            if rgb_patch.dtype != np.uint8:
                rgb_patch = (rgb_patch * 255).astype(np.uint8)
            # 检查尺寸是否有效
            if rgb_patch.shape[0] > 0 and rgb_patch.shape[1] > 0:
                rgb_patch = cv2.resize(rgb_patch, (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                rgb_patch = np.zeros((256, 256, 3), dtype=np.uint8)
            if len(rgb_patch.shape) == 3:
                rgb_patch = np.transpose(rgb_patch, (2, 0, 1))
            # 转换回float32并归一化
            rgb_patch = rgb_patch.astype(np.float32) / 255.0
            
            if len(dsm_patch.shape) == 3:
                dsm_patch = dsm_patch[0]
            # 确保DSM数据类型正确
            if dsm_patch.dtype != np.uint8:
                dsm_patch = (dsm_patch * 255).astype(np.uint8)
            # 检查尺寸是否有效
            if dsm_patch.shape[0] > 0 and dsm_patch.shape[1] > 0:
                dsm_patch = cv2.resize(dsm_patch, (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                dsm_patch = np.zeros((256, 256), dtype=np.uint8)
            if len(dsm_patch.shape) == 2:
                dsm_patch = dsm_patch[np.newaxis, :, :]
            # 转换回float32并归一化
            dsm_patch = dsm_patch.astype(np.float32) / 255.0
            
            # 检查标签尺寸是否有效
            if label_patch.shape[0] > 0 and label_patch.shape[1] > 0:
                label_patch = cv2.resize(label_patch.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            else:
                label_patch = np.zeros((256, 256), dtype=np.uint8)
        
        # 保证连续内存，避免DataLoader collate报storage错误和负步长问题
        # 使用ascontiguousarray确保是C连续的，没有负步长
        rgb_patch = np.ascontiguousarray(rgb_patch, dtype=np.float32)
        dsm_patch = np.ascontiguousarray(dsm_patch, dtype=np.float32)
        label_patch = np.ascontiguousarray(label_patch, dtype=np.uint8)
        
        # 最终尺寸验证
        assert rgb_patch.shape == (3, 256, 256), f"RGB patch shape错误: {rgb_patch.shape}"
        assert dsm_patch.shape == (1, 256, 256), f"DSM patch shape错误: {dsm_patch.shape}"
        assert label_patch.shape == (256, 256), f"Label patch shape错误: {label_patch.shape}"
        
        # 转换为tensor（ascontiguousarray已经确保是C连续的，没有负步长）
        rgb_patch_tensor = torch.from_numpy(rgb_patch).contiguous()
        dsm_patch_tensor = torch.from_numpy(dsm_patch).contiguous()
        label_patch_tensor = torch.from_numpy(label_patch).long().contiguous()
        
        inputs = {
            'rgb': rgb_patch_tensor,
            'dsm': dsm_patch_tensor
        }
        targets = label_patch_tensor
        
        return inputs, targets
    
    def _get_mat_item(self, i):
        """获取MAT数据项"""
        file_dict = self.data_files[0]  # MAT数据集只有一个文件
        
        # 加载数据
        inputs = {}
        for modality, file_path in file_dict.items():
            if modality == 'label':
                continue
            
            try:
                data = self._load_mat_data(file_path)
                if data is not None:
                    if data.ndim == 3:
                        data = data.transpose((2, 0, 1))
                    elif data.ndim == 2:
                        data = data[np.newaxis, :, :]
                    
                    # 归一化
                    data = data.astype(np.float32)
                    if np.max(data) > 1.0:
                        data = data / np.max(data)
                    
                    inputs[modality] = torch.from_numpy(data)
                else:
                    inputs[modality] = torch.randn(1, 256, 256)
            except Exception as e:
                logger.warning(f"加载{modality}数据失败: {e}")
                inputs[modality] = torch.randn(1, 256, 256)
        
        # 加载标签
        try:
            label_data = self._load_mat_data(file_dict['label'])
            if label_data is not None:
                targets = torch.from_numpy(label_data.astype(np.int64))
            else:
                targets = torch.randint(0, self.config['n_classes'], (256, 256))
        except Exception as e:
            logger.warning(f"加载标签数据失败: {e}")
            targets = torch.randint(0, self.config['n_classes'], (256, 256))
        
        return inputs, targets
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True, use_strong=False):
        """
        数据增强
        
        Args:
            *arrays: 要增强的数组（RGB, DSM, Label等）
            flip: 是否允许垂直翻转
            mirror: 是否允许水平镜像
            use_strong: 是否使用强数据增强（旋转、颜色抖动等）
        """
        will_flip, will_mirror = False, False
        
        # 基础增强：翻转和镜像（概率0.5）
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            arr = np.copy(array)  # 初始副本
            
            # 垂直翻转（使用copy避免负步长）
            if will_flip:
                if len(arr.shape) == 2:
                    arr = arr[::-1, :].copy()  # 翻转后立即copy
                elif len(arr.shape) == 3:
                    if arr.shape[0] == 3:  # RGB (C, H, W)
                        arr = arr[:, ::-1, :].copy()
                    else:  # DSM (1, H, W) or (H, W, C)
                        arr = arr[::-1, :, :].copy()
            
            # 水平镜像（使用copy避免负步长）
            if will_mirror:
                if len(arr.shape) == 2:
                    arr = arr[:, ::-1].copy()  # 镜像后立即copy
                elif len(arr.shape) == 3:
                    if arr.shape[0] == 3:  # RGB (C, H, W)
                        arr = arr[:, :, ::-1].copy()
                    else:  # DSM (1, H, W) or (H, W, C)
                        arr = arr[:, :, ::-1].copy()
            
            # 强数据增强（仅在use_strong=True时使用）
            if use_strong:
                # 90度旋转（仅对RGB和DSM，不对Label）
                if len(arrays) > 2:  # 有Label
                    is_label = (array is arrays[-1])  # 最后一个通常是Label
                    if not is_label and random.random() < 0.3:  # 30%概率旋转
                        k = random.randint(1, 3)  # 90, 180, 270度
                        if len(arr.shape) == 2:
                            arr = np.rot90(arr, k).copy()
                        elif len(arr.shape) == 3:
                            if arr.shape[0] == 3:  # RGB (C, H, W)
                                arr = np.transpose(arr, (1, 2, 0))
                                arr = np.rot90(arr, k).copy()
                                arr = np.transpose(arr, (2, 0, 1)).copy()
                            else:  # DSM (1, H, W)
                                arr = arr[0]
                                arr = np.rot90(arr, k).copy()
                                arr = arr[np.newaxis, :, :]
            
            # 最终确保是连续数组（C顺序），避免负步长
            arr = np.ascontiguousarray(arr)
            results.append(arr)
        
        return tuple(results)
    
    def get_random_pos(self, img, window_shape):
        """获取随机位置"""
        w, h = window_shape  # w=width, h=height
        if len(img.shape) == 3:
            H, W = img.shape[1], img.shape[2]  # H=height, W=width
        else:
            H, W = img.shape[0], img.shape[1]  # H=height, W=width
        
        # 确保有足够的空间
        if W < w or H < h:
            # 如果图像太小，从左上角开始
            return 0, min(w, W), 0, min(h, H)
        
        x1 = random.randint(0, W - w)
        x2 = x1 + w
        y1 = random.randint(0, H - h)
        y2 = y1 + h
        
        return x1, x2, y1, y2
    
    def get_safe_random_pos(self, img, window_shape):
        """获取安全的随机位置，确保窗口尺寸正确"""
        w, h = window_shape  # w=width, h=height
        if len(img.shape) == 3:
            H, W = img.shape[1], img.shape[2]  # H=height, W=width
        else:
            H, W = img.shape[0], img.shape[1]  # H=height, W=width
        
        # 确保图像和窗口尺寸有效
        if W <= 0 or H <= 0 or w <= 0 or h <= 0:
            logger.warning(f"无效的尺寸: 图像({W}x{H}), 窗口({w}x{h})")
            return 0, min(w, W), 0, min(h, H)
        
        # 确保图像足够大
        if W < w or H < h:
            # 如果图像太小，返回左上角，后续会resize
            return 0, min(w, W), 0, min(h, H)
        
        # 计算有效的随机范围
        max_x = W - w
        max_y = H - h
        
        # 确保范围有效
        if max_x <= 0 or max_y <= 0:
            return 0, min(w, W), 0, min(h, H)
        
        # 生成随机位置
        x1 = random.randint(0, max_x)
        y1 = random.randint(0, max_y)
        x2 = x1 + w
        y2 = y1 + h
        
        # 最终验证
        if not (x2 - x1 == w and y2 - y1 == h and x1 >= 0 and y1 >= 0 and x2 <= W and y2 <= H):
            logger.warning(f"生成的窗口无效: ({x1},{y1},{x2},{y2}) 图像({W}x{H}) 窗口({w}x{h})")
            return 0, min(w, W), 0, min(h, H)
        
        return x1, x2, y1, y2
    
    @staticmethod
    def convert_from_color(arr_3d, palette=None):
        """RGB颜色编码转换为数值标签 - 完全对齐FTransUNet实现"""
        if palette is None:
            # FTransUNet标准palette（invert_palette）
            palette = {
                (255, 255, 255): 0,  # Impervious surfaces (white)
                (0, 0, 255): 1,      # Buildings (blue)
                (0, 255, 255): 2,    # Low vegetation (cyan)
                (0, 255, 0): 3,      # Trees (green)
                (255, 255, 0): 4,    # Cars (yellow)
                (255, 0, 0): 5,      # Clutter (red)
                (0, 0, 0): 6         # Undefined (black)
            }
        
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
        
        # 使用与FTransUNet完全相同的实现方式（确保兼容性）
        for c, i in palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i
        
        return arr_2d

def create_universal_dataloader(data_path: str, dataset_name: str = 'vaihingen', split: str = 'train',
                               batch_size: int = 4, shuffle: bool = True, num_workers: int = 4,
                               window_size: tuple = (256, 256), stride: int = 32):
    """创建统一数据加载器"""
    dataset = UniversalMultiModalDataset(
        data_path=data_path,
        dataset_name=dataset_name,
        split=split,
        window_size=window_size,
        stride=stride
    )
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

if __name__ == '__main__':
    # 测试统一数据集
    print("测试统一多模态数据集...")
    
    datasets = ['vaihingen', 'houston', 'muufl', 'trento', 'augsburg', 'berlin']
    
    for dataset_name in datasets:
        print(f"\n测试数据集: {dataset_name}")
        try:
            dataloader = create_universal_dataloader(
                data_path='./data',
                dataset_name=dataset_name,
                split='train',
                batch_size=2,
                shuffle=False,
                num_workers=0
            )
            
            print(f"  数据集长度: {len(dataloader.dataset)}")
            
            # 测试一个批次
            for inputs, targets in dataloader:
                print(f"  输入模态: {list(inputs.keys())}")
                for modality, data in inputs.items():
                    print(f"    {modality}: {data.shape}")
                print(f"  标签形状: {targets.shape}")
                break
                
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
    
    print("\n✅ 统一多模态数据集测试完成！")
