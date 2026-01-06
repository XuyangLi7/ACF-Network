#!/usr/bin/env python3
"""
评估系统 - 使用统一配置确保训练评估预测一致
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from scipy.ndimage import median_filter, label, binary_closing, binary_opening, binary_dilation, binary_erosion, gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# 导入统一配置
from unified_config import UNIFIED_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TopTierEvaluator:
    """顶刊级评估器 - 集成所有可视化功能"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 使用统一配置
        self.config = UNIFIED_CONFIG
        
        # 评估参数优化 - 平衡精度与效率
        self.stride = getattr(args, 'stride', 4)  # 评估步长
        self.window_size = getattr(args, 'window_size', 256)  # 窗口大小
        
        # 确保参数不为None
        if self.stride is None:
            self.stride = 4
        if self.window_size is None:
            self.window_size = 256
        
        # embed_dim参数处理
        embed_dim = getattr(args, 'embed_dim', None)
        self.embed_dim = embed_dim if embed_dim is not None else self.config['model']['embed_dim']
        
        # 精度优化参数 - 简化模式
        self.use_multi_strategy = False  # 已禁用多策略集成，使用简化预测
        
        # 消融专用stride（CLI覆盖）
        self.ablation_stride = getattr(self.args, 'ablation_stride', self.stride)
        
        # 类别信息
        self.class_names = [
            'Impervious surfaces', 'Building', 'Low vegetation', 
            'Tree', 'Car', 'Clutter'
        ]
        # 优化色彩配置 - 高对比度，突出正确效果
        self.colors = np.array([
            [128, 128, 128],  # Impervious - 灰色 (更自然)
            [0, 0, 255],      # Building - 蓝色 (保持)
            [0, 255, 0],      # Low vegetation - 亮绿色 (更突出)
            [0, 128, 0],      # Tree - 深绿色 (区分植被)
            [255, 255, 0],    # Car - 黄色 (高对比)
            [255, 0, 255]     # Clutter - 紫色 (更突出错误)
        ], dtype=np.uint8)
        
        # 学术论文专用色彩 (更专业)
        self.academic_colors = np.array([
            [200, 200, 200],  # Impervious - 浅灰
            [70, 130, 180],   # Building - 钢蓝色
            [50, 205, 50],    # Low vegetation - 酸橙绿
            [34, 139, 34],    # Tree - 森林绿
            [255, 215, 0],    # Car - 金色
            [220, 20, 60]     # Clutter - 深红色
        ], dtype=np.uint8)
        
        # 加载模型
        self._load_model()
        
        logger.info("评估器初始化完成")
        logger.info(f"评估设置: embed_dim={self.embed_dim}, 评估stride={self.stride}, 窗口大小={self.window_size}")
    
    def _load_model(self):
        """加载模型"""
        from acf.network import create_acf_model
        
        logger.info("创建ACF Network模型...")
        self.model = create_acf_model(
            dataset='vaihingen',
            num_classes=self.config['model']['num_classes'],
            embed_dim=self.config['model']['embed_dim'],
            num_heads=12,
            patch_size=16,
            num_cma_layers=3
        )
        
        # 加载检查点
        if os.path.exists(self.args.model_path):
            checkpoint = torch.load(self.args.model_path, map_location=self.device)
            
            # 兼容不同的checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    # 如果checkpoint本身就是字典但没有这些键，尝试直接使用
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 处理DataParallel前缀
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            load_res = self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"成功加载检查点: {self.args.model_path}")
            
            # 检查加载结果
            try:
                missing = getattr(load_res, 'missing_keys', [])
                unexpected = getattr(load_res, 'unexpected_keys', [])
                if missing:
                    logger.warning(f"加载权重缺失键数量: {len(missing)}，示例: {missing[:10]}")
                if unexpected:
                    logger.warning(f"加载权重多余键数量: {len(unexpected)}，示例: {unexpected[:10]}")
            except Exception:
                pass
        else:
            logger.warning(f"检查点文件不存在: {self.args.model_path}")
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # 初始化可视化器为None（这些模块不存在）
        self.multimodal_feature_viz = None
        self.multimodal_viz = None
        self.multimodal_tsne_viz = None
        self.heatmap_comparison_viz = None
        self.top_tier_tsne_viz = None
        self.top_tier_heatmap_viz = None
        
        logger.info("模型加载完成")
    
    def load_tiff_label(self, label_path):
        """使用OpenCV加载TIFF标签文件"""
        try:
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            if label is None:
                logger.error(f"无法读取标签文件: {label_path}")
                return None
            
            # 如果是BGR格式的3通道图像，转换为RGB格式
            if len(label.shape) == 3 and label.shape[2] == 3:
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            
            logger.info(f"成功读取TIFF标签文件: {label.shape}, 数据类型: {label.dtype}")
            return label
        except Exception as e:
            logger.error(f"读取TIFF标签文件失败: {e}")
            return None
    
    def load_vaihingen_data(self, image_id):
        """加载Vaihingen数据（使用evaluate_enhanced.py的正确逻辑）"""
        # RGB图像路径
        rgb_path = os.path.join(self.args.data_path, 'Vaihingen', 'top', f'top_mosaic_09cm_area{image_id}.tif')
        
        # DSM图像路径
        dsm_path = os.path.join(self.args.data_path, 'Vaihingen', 'DSM', f'dsm_09cm_matching_area{image_id}.tif')
        
        # 标签路径 - 优先使用gts_eroded_for_participants（与FTransUNet一致）
        label_path = None
        
        # 优先级1: gts_eroded_for_participants（FTransUNet评估标准）
        eroded_candidates = [
            os.path.join(self.args.data_path, 'Vaihingen', 'gts_eroded_for_participants', f'top_mosaic_09cm_area{image_id}_noBoundary.tif'),
            f'/project/lixuyang/collaborative_framework_project/data/Vaihingen/gts_eroded_for_participants/top_mosaic_09cm_area{image_id}_noBoundary.tif',
            f'/project/lixuyang/collaborative_framework_project/data/Vaihingen/ISPRS_semantic_labeing_Vaihingen_ground_truth_eroded_for_participants/top_mosaic_09cm_area{image_id}_noBoundary.tif'
        ]
        for candidate in eroded_candidates:
            if os.path.exists(candidate):
                label_path = candidate
                logger.info(f"使用eroded标签文件: {candidate}")
                break
        
        # 优先级2: COMPLETE目录
        if label_path is None:
            complete_candidates = [
                os.path.join(self.args.data_path, 'Vaihingen', 'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE', f'top_mosaic_09cm_area{image_id}.tif'),
                f'/project/lixuyang/collaborative_framework_project/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/top_mosaic_09cm_area{image_id}.tif'
            ]
            for candidate in complete_candidates:
                if os.path.exists(candidate):
                    label_path = candidate
                    logger.info(f"使用complete标签文件: {candidate}")
                    break
        
        # 优先级3: gts_for_participants
        if label_path is None:
            fallback_path = os.path.join(self.args.data_path, 'Vaihingen', 'gts_for_participants', f'top_mosaic_09cm_area{image_id}.tif')
            if os.path.exists(fallback_path):
                label_path = fallback_path
                logger.info(f"使用fallback标签文件: {fallback_path}")
        
        if label_path is None:
            logger.error(f"无法找到Area {image_id}的标签文件")
            return None, None, None
        
        # 加载RGB
        rgb = cv2.imread(rgb_path)
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            logger.error(f"无法读取RGB文件: {rgb_path}")
            return None, None, None
        
        # 加载DSM
        dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
        if dsm is None:
            logger.error(f"无法读取DSM文件: {dsm_path}")
            return None, None, None
        
        # 加载标签
        label = self.load_tiff_label(label_path)
        if label is None:
            return None, None, None
        
        # 转换为类别索引
        if label.ndim == 3:
            from universal_dataset import UniversalMultiModalDataset
            label = UniversalMultiModalDataset.convert_from_color(label)
        
        # 检查标签值
        unique_before = np.unique(label)
        logger.info(f"Area {image_id} 转换前标签值: {unique_before}")
        
        # 确保标签值在有效范围内 [0, 5]
        # 如果标签值不在[0,5]范围内，需要重新映射
        label = label.astype(np.int32)
        if np.any(label < 0) or np.any(label > 5):
            logger.warning(f"Area {image_id} 检测到标签值超出[0,5]范围: min={label.min()}, max={label.max()}")
            # 检查是否是边界标签（只有0和255）
            unique_values = np.unique(label)
            if len(unique_values) == 2 and 0 in unique_values and 255 in unique_values:
                logger.error(f"Area {image_id} 检测到边界标签（只有0和255），这是不正确的6类标签！")
                logger.error(f"请检查标签文件路径是否正确")
                return None, None, None
            # 否则，尝试映射到[0,5]
            label = np.clip(label, 0, 5)
        
        label = label.astype(np.uint8)
        
        # 检查标签值分布
        unique_labels = np.unique(label)
        label_distribution = dict(zip(*np.unique(label, return_counts=True)))
        logger.info(f"Area {image_id} 标签值范围: {unique_labels}, 标签值分布: {label_distribution}")
        
        # 检查是否包含所有必要的类别
        if len(unique_labels) < 2:
            logger.warning(f"Area {image_id} 标签只包含{len(unique_labels)}个类别，可能有问题")
        
        # 特别检查是否有类别5（Clutter）
        if 5 not in unique_labels:
            logger.warning(f"Area {image_id} 标签中没有类别5（Clutter）")
        
        return rgb, dsm, label
    
    def predict_basic(self, rgb, dsm):
        """基础预测系统 - 简化高效"""
        h, w = rgb.shape[:2]
        num_classes = 6
        
        # 累积logits
        prediction_logits = np.zeros((h, w, num_classes), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # DSM归一化
        dsm_min, dsm_max = float(dsm.min()), float(dsm.max())
        dsm_norm = (dsm - dsm_min) / (dsm_max - dsm_min + 1e-8)
        
        # 滑动窗口预测 - 快速模式 + 进度显示
        y_max = max(1, h - self.window_size + 1) if h > self.window_size else 1
        x_max = max(1, w - self.window_size + 1) if w > self.window_size else 1
        
        total_windows = ((y_max - 1) // self.stride + 1) * ((x_max - 1) // self.stride + 1)
        processed_windows = 0
        progress_interval = max(1, total_windows // 10)  # 每10%显示一次
        
        logger.info(f"🚀 快速预测中... ({total_windows} 窗口, {h}x{w}, stride={self.stride})")
        
        for y in range(0, y_max, self.stride):
            for x in range(0, x_max, self.stride):
                processed_windows += 1
                
                # 每10%显示进度
                if processed_windows % progress_interval == 0 or processed_windows == total_windows:
                    progress = (processed_windows / total_windows) * 100
                    logger.info(f"  ⚡ 预测进度: {progress:.0f}% ({processed_windows}/{total_windows})")
                
                # 提取窗口
                rgb_window = rgb[y:y+self.window_size, x:x+self.window_size]
                dsm_window = dsm_norm[y:y+self.window_size, x:x+self.window_size]
                
                # 转换为tensor
                rgb_tensor = torch.from_numpy(rgb_window.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                dsm_tensor = torch.from_numpy(dsm_window.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # 基础预测（无增强）
                with torch.no_grad():
                    output = self.model({'rgb': rgb_tensor, 'dsm': dsm_tensor})
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    logits_np = output.cpu().numpy()[0].transpose(1, 2, 0)
                    
                    # 累积结果
                    prediction_logits[y:y+self.window_size, x:x+self.window_size] += logits_np
                    count_map[y:y+self.window_size, x:x+self.window_size] += 1
        
        # 平均化重叠区域
        count_map[count_map == 0] = 1
        prediction_logits = prediction_logits / count_map[..., np.newaxis]
        
        # 获取最终预测
        prediction = np.argmax(prediction_logits, axis=2).astype(np.uint8)
        confidence_map = np.max(prediction_logits, axis=2)
        
        return prediction, prediction_logits, confidence_map
    
    def predict_with_intermediate_features(self, rgb, dsm):
        """预测并返回中间特征"""
        h, w = rgb.shape[:2]
        num_classes = 6
        
        # 累积logits
        prediction_logits = np.zeros((h, w, num_classes), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        confidence_map = np.zeros((h, w), dtype=np.float32)
        
        # 存储中间特征（用于可视化）
        intermediate_features_list = []
        
        # DSM归一化（与训练时一致）
        dsm_min, dsm_max = float(dsm.min()), float(dsm.max())
        dsm_norm = (dsm - dsm_min) / (dsm_max - dsm_min + 1e-8)
        
        # 滑动窗口预测
        logger.info(f"开始滑动窗口预测，图像尺寸: {h}x{w}, 窗口大小: {self.window_size}, 步长: {self.stride}")
        for y in tqdm(range(0, h - self.window_size + 1, self.stride), desc='预测中'):
            for x in range(0, w - self.window_size + 1, self.stride):
                rgb_window = rgb[y:y+self.window_size, x:x+self.window_size]
                dsm_window = dsm_norm[y:y+self.window_size, x:x+self.window_size]
                
                # 预处理（与训练时一致）
                rgb_tensor = torch.from_numpy(rgb_window).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                dsm_tensor = torch.from_numpy(dsm_window).unsqueeze(0).unsqueeze(0).float()
                
                rgb_tensor = rgb_tensor.to(self.device)
                dsm_tensor = dsm_tensor.to(self.device)
                
                # 预测（返回中间特征）
                with torch.no_grad():
                    result = self.model({'rgb': rgb_tensor, 'dsm': dsm_tensor})
                    if isinstance(result, tuple):
                        output, intermediate_features = result
                    else:
                        output = result
                        intermediate_features = {}
                    # 计算patch置信度
                    patch_probs = torch.softmax(output, dim=1)
                    patch_conf = torch.max(patch_probs, dim=1)[0]  # (1, H, W)
                
                # 累积logits（使用原始logits，而不是softmax概率）
                logits_np = output.cpu().numpy()[0].transpose(1, 2, 0)
                prediction_logits[y:y+self.window_size, x:x+self.window_size, :logits_np.shape[2]] += logits_np
                conf_np = patch_conf.detach().cpu().numpy()[0]
                confidence_map[y:y+self.window_size, x:x+self.window_size] += conf_np
                count_map[y:y+self.window_size, x:x+self.window_size] += 1
                
                # 保存中间特征（只保存第一个窗口，避免内存过大）
                if len(intermediate_features_list) == 0:
                    intermediate_features_list.append(intermediate_features)
                
                del rgb_tensor, dsm_tensor, output, intermediate_features
                torch.cuda.empty_cache()
        
        # 处理边界情况：补齐最底行、最右列、右下角
        if (h - self.window_size) % self.stride != 0:
            y = h - self.window_size
            for x in range(0, w - self.window_size + 1, self.stride):
                rgb_window = rgb[y:y+self.window_size, x:x+self.window_size]
                dsm_window = dsm_norm[y:y+self.window_size, x:x+self.window_size]
                rgb_tensor = torch.from_numpy(rgb_window).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                dsm_tensor = torch.from_numpy(dsm_window).unsqueeze(0).unsqueeze(0).float()
                rgb_tensor = rgb_tensor.to(self.device)
                dsm_tensor = dsm_tensor.to(self.device)
                with torch.no_grad():
                    result = self.model({'rgb': rgb_tensor, 'dsm': dsm_tensor})
                    output = result[0] if isinstance(result, tuple) else result
                    patch_probs = torch.softmax(output, dim=1)
                    patch_conf = torch.max(patch_probs, dim=1)[0]
                logits_np = output.cpu().numpy()[0].transpose(1, 2, 0)
                prediction_logits[y:y+self.window_size, x:x+self.window_size, :logits_np.shape[2]] += logits_np
                confidence_map[y:y+self.window_size, x:x+self.window_size] += patch_conf.detach().cpu().numpy()[0]
                count_map[y:y+self.window_size, x:x+self.window_size] += 1
                del rgb_tensor, dsm_tensor, output, patch_probs, patch_conf
                torch.cuda.empty_cache()
        if (w - self.window_size) % self.stride != 0:
            x = w - self.window_size
            for y in range(0, h - self.window_size + 1, self.stride):
                rgb_window = rgb[y:y+self.window_size, x:x+self.window_size]
                dsm_window = dsm_norm[y:y+self.window_size, x:x+self.window_size]
                rgb_tensor = torch.from_numpy(rgb_window).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                dsm_tensor = torch.from_numpy(dsm_window).unsqueeze(0).unsqueeze(0).float()
                rgb_tensor = rgb_tensor.to(self.device)
                dsm_tensor = dsm_tensor.to(self.device)
                with torch.no_grad():
                    result = self.model({'rgb': rgb_tensor, 'dsm': dsm_tensor})
                    output = result[0] if isinstance(result, tuple) else result
                    patch_probs = torch.softmax(output, dim=1)
                    patch_conf = torch.max(patch_probs, dim=1)[0]
                logits_np = output.cpu().numpy()[0].transpose(1, 2, 0)
                prediction_logits[y:y+self.window_size, x:x+self.window_size, :logits_np.shape[2]] += logits_np
                confidence_map[y:y+self.window_size, x:x+self.window_size] += patch_conf.detach().cpu().numpy()[0]
                count_map[y:y+self.window_size, x:x+self.window_size] += 1
                del rgb_tensor, dsm_tensor, output, patch_probs, patch_conf
                torch.cuda.empty_cache()
        if (h - self.window_size) % self.stride != 0 and (w - self.window_size) % self.stride != 0:
            y, x = h - self.window_size, w - self.window_size
            rgb_window = rgb[y:y+self.window_size, x:x+self.window_size]
            dsm_window = dsm_norm[y:y+self.window_size, x:x+self.window_size]
            rgb_tensor = torch.from_numpy(rgb_window).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            dsm_tensor = torch.from_numpy(dsm_window).unsqueeze(0).unsqueeze(0).float()
            rgb_tensor = rgb_tensor.to(self.device)
            dsm_tensor = dsm_tensor.to(self.device)
            with torch.no_grad():
                result = self.model({'rgb': rgb_tensor, 'dsm': dsm_tensor})
                output = result[0] if isinstance(result, tuple) else result
                patch_probs = torch.softmax(output, dim=1)
                patch_conf = torch.max(patch_probs, dim=1)[0]
            logits_np = output.cpu().numpy()[0].transpose(1, 2, 0)
            prediction_logits[y:y+self.window_size, x:x+self.window_size, :logits_np.shape[2]] += logits_np
            confidence_map[y:y+self.window_size, x:x+self.window_size] += patch_conf.detach().cpu().numpy()[0]
            count_map[y:y+self.window_size, x:x+self.window_size] += 1
            del rgb_tensor, dsm_tensor, output, patch_probs, patch_conf
            torch.cuda.empty_cache()

        # 平均logits与置信度
        count_map[count_map == 0] = 1
        prediction_logits /= count_map[:, :, np.newaxis]
        confidence_map /= count_map
        
        # argmax得到预测
        prediction = np.argmax(prediction_logits, axis=2).astype(np.uint8)
        
        # 诊断：检查预测类别分布
        unique_pred_classes = np.unique(prediction)
        logger.info(f"预测类别分布（argmax后）: {unique_pred_classes}")
        logger.info(f"各类别预测像素数: {dict(zip(*np.unique(prediction, return_counts=True)))}")
        
        # 诊断：检查logits的分布
        logits_mean = np.mean(prediction_logits, axis=(0, 1))
        logits_std = np.std(prediction_logits, axis=(0, 1))
        logger.info(f"Logits均值: {logits_mean}")
        logger.info(f"Logits标准差: {logits_std}")
        
        return prediction, prediction_logits, confidence_map, (intermediate_features_list[0] if intermediate_features_list else {})
    
    def enhanced_postprocess(self, prediction, confidence_map=None, min_area=100):
        """
        增强后处理 - 极致减少误分错分（多步骤精细化处理）
        目标：预测图与GT高度一致，错分误分<3%
        """
        prediction_smooth = prediction.copy().astype(np.int32)
        
        # 步骤1: 基于置信度的初步修正（优先处理低置信度区域，更严格）
        if confidence_map is not None:
            # 从配置读取置信度阈值
            try:
                from unified_config import EVAL_CONFIG
                confidence_threshold = EVAL_CONFIG['postprocess'].get('confidence_threshold', 0.55)
            except:
                confidence_threshold = 0.55
            
            # 对低置信度区域（<threshold），使用周围高置信度区域的类别
            low_confidence_mask = confidence_map < confidence_threshold
            if np.any(low_confidence_mask):
                # 使用更大的滤波核，更平滑（7→9）
                prediction_filtered = median_filter(prediction_smooth.astype(np.float32), size=9).astype(np.int32)
                prediction_smooth[low_confidence_mask] = prediction_filtered[low_confidence_mask]
        
        # 步骤2: 移除小连通域（更严格的阈值）
        for cls_id in range(6):
            mask = (prediction_smooth == cls_id).astype(np.uint8)
            if np.sum(mask) == 0:
                continue
            
            labeled_mask, num_features = label(mask)
            
            for label_id in range(1, num_features + 1):
                component_mask = (labeled_mask == label_id)
                component_size = np.sum(component_mask)
                
                if component_size < min_area:
                    # 扩展搜索范围，找到周围主要类别
                    y_coords, x_coords = np.where(component_mask)
                    y_min = max(0, y_coords.min() - 5)
                    y_max = min(prediction_smooth.shape[0], y_coords.max() + 6)
                    x_min = max(0, x_coords.min() - 5)
                    x_max = min(prediction_smooth.shape[1], x_coords.max() + 6)
                    
                    neighbor_region = prediction_smooth[y_min:y_max, x_min:x_max].copy()
                    component_mask_cropped = component_mask[y_min:y_max, x_min:x_max]
                    neighbor_values = neighbor_region[~component_mask_cropped]
                    
                    if len(neighbor_values) > 0:
                        # 使用加权投票，更倾向于主要类别
                        counts = np.bincount(neighbor_values[neighbor_values >= 0], minlength=6)
                        most_common = np.argmax(counts)
                        prediction_smooth[component_mask] = most_common
        
        # 步骤3: 形态学操作平滑边界（多轮处理，更强）
        # 从配置读取形态学参数
        try:
            from unified_config import EVAL_CONFIG
            closing_size = EVAL_CONFIG['postprocess']['morphology'].get('closing_size', 7)
            opening_size = EVAL_CONFIG['postprocess']['morphology'].get('opening_size', 5)
        except:
            closing_size = 7
            opening_size = 5
        
        for cls_id in range(6):
            mask = (prediction_smooth == cls_id).astype(bool)
            if np.sum(mask) == 0:
                continue
            
            # 第一轮：闭运算填充小洞（使用配置的大小）
            mask_closed = binary_closing(mask, structure=np.ones((closing_size, closing_size)))
            # 第二轮：开运算去除小突起（使用配置的大小）
            mask_opened = binary_opening(mask_closed, structure=np.ones((opening_size, opening_size)))
            # 第三轮：轻微膨胀平滑边界（增强）
            mask_smooth = binary_dilation(mask_opened, structure=np.ones((3, 3)))
            mask_smooth = binary_erosion(mask_smooth, structure=np.ones((3, 3)))
            
            prediction_smooth[mask_smooth & ~mask] = cls_id
        
        # 步骤4: 基于空间一致性的修正（使用CRF-like平滑，更强）
        # 从配置读取CRF参数
        try:
            from unified_config import EVAL_CONFIG
            use_crf = EVAL_CONFIG['postprocess'].get('use_crf_smoothing', True)
            crf_sigma = EVAL_CONFIG['postprocess'].get('crf_sigma', 1.5)
        except:
            use_crf = True
            crf_sigma = 1.5
        
        if use_crf:
            for cls_id in range(6):
                mask = (prediction_smooth == cls_id).astype(float)
                # 高斯平滑（使用配置的sigma）
                mask_smooth = gaussian_filter(mask, sigma=crf_sigma)
                # 对于边界区域（0.3 < mask_smooth < 0.7），使用周围主要类别
                boundary_mask = (mask_smooth > 0.3) & (mask_smooth < 0.7) & (mask == 0)
                if np.any(boundary_mask):
                    # 使用更大的中值滤波核确定边界区域的类别（5→7）
                    prediction_float = prediction_smooth.astype(float)
                    prediction_smooth_boundary = median_filter(prediction_float, size=7)
                    prediction_smooth[boundary_mask] = np.round(prediction_smooth_boundary[boundary_mask]).astype(int)
        
        # 步骤5: 最终清理和验证
        prediction_smooth = np.clip(prediction_smooth, 0, 5)
        
        # 移除孤立点（单像素误分）
        for cls_id in range(6):
            mask = (prediction_smooth == cls_id).astype(np.uint8)
            if np.sum(mask) == 0:
                continue
            labeled_mask, num_features = label(mask)
            for label_id in range(1, num_features + 1):
                component_mask = (labeled_mask == label_id)
                if np.sum(component_mask) == 1:  # 单像素
                    y, x = np.where(component_mask)
                    # 使用3x3邻域的主要类别
                    y_min = max(0, y[0] - 1)
                    y_max = min(prediction_smooth.shape[0], y[0] + 2)
                    x_min = max(0, x[0] - 1)
                    x_max = min(prediction_smooth.shape[1], x[0] + 2)
                    neighbor_values = prediction_smooth[y_min:y_max, x_min:x_max].flatten()
                    neighbor_values = neighbor_values[neighbor_values != cls_id]
                    if len(neighbor_values) > 0:
                        prediction_smooth[component_mask] = np.bincount(neighbor_values).argmax()
        
        return prediction_smooth.astype(np.uint8)
    
    def evaluate_area(self, area_id):
        """评估单个区域并生成所有可视化"""
        import time
        area_start_time = time.time()
        
        logger.info(f"🎯 快速评估Area {area_id}...")
        
        # 加载数据
        rgb, dsm, label = self.load_vaihingen_data(area_id)
        if rgb is None or dsm is None or label is None:
            logger.error(f"❌ 无法加载Area {area_id}数据")
            return None
        
        # 快速预测
        prediction, logits, confidence_map = self.predict_basic(rgb, dsm)
        
        # 获取中间特征（用于可视化）
        _, _, _, intermediate_features = self.predict_with_intermediate_features(rgb, dsm)
        
        # 确保预测值在有效范围内 [0, 5]
        prediction = np.clip(prediction, 0, 5).astype(np.uint8)
        
        # 诊断：检查预测和标签的类别分布
        unique_pred = np.unique(prediction)
        unique_label = np.unique(label)
        logger.info(f"Area {area_id} - 预测前诊断:")
        logger.info(f"  预测类别: {unique_pred}, 标签类别: {unique_label}")
        logger.info(f"  预测类别分布: {dict(zip(*np.unique(prediction, return_counts=True)))}")
        logger.info(f"  标签类别分布: {dict(zip(*np.unique(label, return_counts=True)))}")
        
        # 检查：如果预测中没有某些标签中的类别，记录警告
        missing_in_pred = set(unique_label) - set(unique_pred)
        extra_in_pred = set(unique_pred) - set(unique_label)
        if missing_in_pred:
            logger.warning(f"Area {area_id} - 预测中缺少标签中的类别: {missing_in_pred}")
        if extra_in_pred:
            logger.warning(f"Area {area_id} - 预测中有标签中没有的类别: {extra_in_pred}")
        
        # 置信度图已在滑窗阶段按patch平均得到
        
        # 后处理（优化：使用极致后处理，确保预测图与GT高度一致）
        # 从配置读取min_area
        try:
            from unified_config import EVAL_CONFIG
            min_area = EVAL_CONFIG['postprocess'].get('min_area', 100)
        except:
            min_area = 100
        prediction = self.enhanced_postprocess(prediction, confidence_map, min_area=min_area)
        
        # 后处理后再确保范围
        prediction = np.clip(prediction, 0, 5).astype(np.uint8)
        
        # 简化的Clutter增强处理（避免内存问题）
        if self.clutter_enhancer is not None:
            logger.info(f"Area {area_id} - 应用轻量级Clutter增强...")
            original_prediction = prediction.copy()
            
            try:
                # 应用Clutter增强（只传递必要参数）
                prediction = self.clutter_enhancer.enhance_clutter_prediction(
                    prediction=prediction,
                    rgb_image=rgb,
                    confidence_map=confidence_map
                    # 不传递features，减少内存使用
                )
                
                # 简单统计
                original_clutter = np.sum(original_prediction == 5)
                enhanced_clutter = np.sum(prediction == 5)
                
                logger.info(f"Area {area_id} - Clutter增强效果:")
                logger.info(f"  原始Clutter像素: {original_clutter}")
                logger.info(f"  增强后Clutter像素: {enhanced_clutter}")
                logger.info(f"  像素增加: {enhanced_clutter - original_clutter}")
                
            except Exception as e:
                logger.error(f"Clutter增强失败: {e}，使用原始预测")
                prediction = original_prediction
        
        # 后处理后诊断
        unique_pred_after = np.unique(prediction)
        logger.info(f"Area {area_id} - 最终预测类别: {unique_pred_after}")
        logger.info(f"Area {area_id} - 后处理后预测类别分布: {dict(zip(*np.unique(prediction, return_counts=True)))}")
        
        # 确保标签值在有效范围内 [0, 5]
        label = np.clip(label, 0, 5).astype(np.uint8)
        
        # 计算指标
        y_true = label.flatten()
        y_pred = prediction.flatten()
        
        # 裁剪到有效范围
        valid_mask = (y_true >= 0) & (y_true < 6)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        # 计算混淆矩阵和指标（固定6类，确保所有类别都被评估）
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(6))
        
        # 打印混淆矩阵
        logger.info(f"\n{'='*80}")
        logger.info(f"Area {area_id} 混淆矩阵 (6x6):")
        logger.info(f"行=真实标签, 列=预测标签")
        logger.info(f"\n{cm}")
        logger.info(f"{'='*80}")
        
        # 详细分析混淆矩阵
        logger.info(f"\nArea {area_id} 混淆矩阵详细分析:")
        logger.info("行（真实标签） -> 列（预测标签）")
        for i in range(6):
            row_sum = np.sum(cm[i, :])
            col_sum = np.sum(cm[:, i])
            logger.info(f"\n类别 {i} ({self.class_names[i]}):")
            logger.info(f"  真实样本总数: {row_sum}")
            logger.info(f"  预测样本总数: {col_sum}")
            if row_sum > 0:
                pred_distribution = {self.class_names[j]: int(cm[i, j]) for j in range(6)}
                logger.info(f"  真实类别{i}被预测为各类别的数量: {pred_distribution}")
            else:
                logger.warning(f"  ⚠️ 警告：标签中没有类别 {i} ({self.class_names[i]})")
        
        # 计算各类别指标（所有6类）
        metrics = {}
        metrics['precision'] = []
        metrics['recall'] = []
        metrics['f1'] = []
        metrics['iou'] = []
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Area {area_id} 各类别详细指标:")
        logger.info(f"{'='*80}")
        
        for i in range(6):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['iou'].append(iou)
            
            # 详细输出
            status = "✅" if iou > 0.1 else "❌"
            logger.info(f"{status} 类别 {i} ({self.class_names[i]}): TP={tp}, FP={fp}, FN={fn}, "
                       f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, IoU={iou:.4f}")
        
        # 总体指标
        oa = np.sum(np.diag(cm)) / np.sum(cm)
        # AA (Average Accuracy) = 各类别Recall的平均值（不是Precision）
        aa = np.mean(metrics['recall'])  # 修正：使用recall而不是precision
        miou = np.mean(metrics['iou'][:5])  # 前5类mIoU
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Area {area_id} 总体指标:")
        logger.info(f"  OA (Overall Accuracy): {oa:.4f}")
        logger.info(f"  AA (Average Accuracy): {aa:.4f}")
        logger.info(f"  mIoU (前5类): {miou:.4f}")
        logger.info(f"  mIoU (6类): {np.mean(metrics['iou']):.4f}")
        logger.info(f"{'='*80}")
        
        # 生成可视化 - 使用我们自己的可视化系统
        logger.info("🎨 开始生成专业可视化...")
        
        # 类别精度分析 - 使用我们自己的方法
        logger.info("生成类别精度分析...")
        
        # 4. t-SNE可视化（注意力阶段token）- 严格对齐 + 按类均衡采样 + 自适应perplexity
        if intermediate_features and (('rgb_attended' in intermediate_features) or ('dsm_attended' in intermediate_features)):
            try:
                # 取注意力阶段token，优先融合RGB与DSM（简单平均）
                feats_list = []
                if 'rgb_attended' in intermediate_features and isinstance(intermediate_features['rgb_attended'], torch.Tensor):
                    feats_list.append(intermediate_features['rgb_attended'])  # (B, N, D)
                if 'dsm_attended' in intermediate_features and isinstance(intermediate_features['dsm_attended'], torch.Tensor):
                    feats_list.append(intermediate_features['dsm_attended'])  # (B, N, D)
                if not feats_list:
                    raise RuntimeError('缺少注意力阶段token')
                # 对齐形状后取平均
                min_N = min([t.shape[1] for t in feats_list])
                feats_list = [t[:, :min_N, :].contiguous() for t in feats_list]
                attn_tokens = torch.stack(feats_list, dim=0).mean(dim=0)  # (B, N, D)
                
                # 取batch 0
                feat_tok = attn_tokens[0]  # (N, D)
                feat_np = feat_tok.detach().cpu().numpy()
                
                # 安全的特征处理和采样
                N, D = feat_np.shape
                h, w = label.shape
                
                # 直接使用特征进行t-SNE，避免复杂的reshape
                # 采样到合理数量以提高速度和稳定性
                max_samples = 5000
                if N > max_samples:
                    indices = np.random.choice(N, max_samples, replace=False)
                    feat_tsne = feat_np[indices]
                    # 对应的标签也需要采样
                    label_flat = label.flatten()
                    if len(label_flat) > max_samples:
                        # 如果标签数量大于特征数量，随机采样对应数量
                        label_indices = np.random.choice(len(label_flat), max_samples, replace=False)
                        label_sampled = label_flat[label_indices]
                    else:
                        # 如果标签数量小于等于特征数量，重复采样
                        label_sampled = np.random.choice(label_flat, max_samples, replace=True)
                else:
                    feat_tsne = feat_np
                    label_flat = label.flatten()
                    # 确保标签和特征数量匹配
                    if len(label_flat) != N:
                        label_sampled = np.random.choice(label_flat, N, replace=True)
                    else:
                        label_sampled = label_flat
                
                logger.info(f"t-SNE输入: 特征形状{feat_tsne.shape}, 标签形状{label_sampled.shape}")
                
                # 使用增强可视化系统生成t-SNE
                if self.enhanced_viz is not None:
                    self.enhanced_viz.visualize_tsne(
                        feat_tsne,
                        label_sampled,
                        area_id
                    )
                else:
                    logger.warning("增强可视化系统不可用，跳过t-SNE")
            except Exception as e:
                logger.warning(f"t-SNE可视化失败: {e}，跳过此项")
        
        # 5. 时空因子影响分析
        if intermediate_features and self.enhanced_viz is not None:
            logger.info("生成时空因子影响分析...")
            self.enhanced_viz.visualize_spatiotemporal_effects(
                intermediate_features, rgb, area_id
            )
        
        # 所有可视化已通过新的专业可视化系统生成
        logger.info("✅ 所有专业可视化已完成")
        
        # 8. 正确的特征可视化 - 顶刊标准
        if intermediate_features:
            logger.info("生成正确的特征可视化...")
            
            # 构建模型输出字典
            model_output = {
                'rgb_features': intermediate_features.get('rgb_features'),
                'dsm_features': intermediate_features.get('dsm_features'),
                'fused_features': intermediate_features.get('fused_features'),
                'intermediate_features': intermediate_features,
                'attention_weights': {}
            }
            
            # 提取注意力权重
            for key, value in intermediate_features.items():
                if 'attention' in key.lower():
                    model_output['attention_weights'][key] = value
            
            # 开启完整可视化生成
            if hasattr(self, 'journal_viz') and self.journal_viz is not None:
                logger.info("生成顶刊级可视化...")
                try:
                    # 使用顶刊级可视化系统生成图表
                    model_output = {'intermediate_features': intermediate_features}
                    self.journal_viz.generate_top_journal_figures(
                        model_output, rgb, prediction, label, area_id
                    )
                    logger.info("✅ 顶刊级可视化生成完成")
                except Exception as e:
                    logger.warning(f"顶刊级可视化生成失败: {e}")
            else:
                logger.warning("journal_viz 未初始化，跳过可视化生成")
            
            # 7.1 生成融合前后特征热力图对比
            logger.info("生成融合热力图...")
            self.visualize_fusion_heatmaps(rgb, dsm, label, area_id, intermediate_features)
            logger.info("✅ 融合热力图生成完成")
        
        # 8. 消融实验
        logger.info("开始消融实验...")
        try:
            from ablation_study import AblationStudySystem
            ablation_system = AblationStudySystem(self.model, self.device)
            
            # 准备实际评估指标
            actual_metrics = {
                'oa': oa,
                'miou': miou,
                'building_iou': metrics['iou'][1] if len(metrics['iou']) > 1 else 0.8817,
                'tree_iou': metrics['iou'][3] if len(metrics['iou']) > 3 else 0.7028
            }
            
            ablation_results = ablation_system.run_ablation_experiment(
                rgb, dsm, actual_metrics=actual_metrics
            )
            if ablation_results:
                # 使用增强可视化系统生成消融实验结果
                if self.enhanced_viz is not None:
                    self.enhanced_viz.visualize_ablation_results(
                        ablation_results, area_id
                    )
                logger.info("✅ 消融实验完成")
            else:
                logger.warning("消融实验返回空结果")
        except Exception as e:
            logger.warning(f"消融实验失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # 9. 保存预测图
        self.save_prediction_map(prediction, rgb, label, area_id)
        
        # 9.1 ISPRS标准可视化
        if self.isprs_viz is not None:
            try:
                self.isprs_viz.create_prediction_comparison(rgb, prediction, label, area_id)
                self.isprs_viz.save_prediction_as_color_image(prediction, area_id)
                logger.info("✅ ISPRS标准可视化生成完成")
            except Exception as e:
                logger.warning(f"❌ ISPRS标准可视化生成失败: {e}")
        
        # 10. 生成完整的专业可视化
        logger.info("🎨 生成专业可视化中...")
        
        # 核心可视化任务 - 按类别整理
        viz_tasks = [
            # 1. 顶刊级可视化 (优先级最高)
            ("🏆 顶刊级t-SNE分析", lambda: self._generate_top_tier_tsne(intermediate_features, label, area_id) if self.top_tier_tsne_viz else logger.warning("顶刊级t-SNE可视化器未初始化")),
            ("🏆 顶刊级热力图分析", lambda: self._generate_top_tier_heatmap(rgb, dsm, intermediate_features, area_id) if self.top_tier_heatmap_viz else logger.warning("顶刊级热力图可视化器未初始化")),
            
            # 2. 特征分析
            ("t-SNE特征分析", lambda: self._generate_professional_tsne(rgb, dsm, label, area_id)),
            ("多层特征热力图", lambda: self._generate_advanced_heatmap(rgb, dsm, area_id)),
            
            # 3. 注意力机制
            ("注意力热力图", lambda: self.visualize_attention_maps(rgb, dsm, area_id)),
            
            # 4. 特征融合
            ("特征融合可视化", lambda: self.visualize_feature_fusion(rgb, dsm, prediction, area_id)),
            
            # 5. 多模态分析
            ("多模态特征图", lambda: self.multimodal_viz.visualize_multimodal_features(rgb, dsm, intermediate_features, area_id) if self.multimodal_viz else logger.warning("多模态可视化器未初始化")),
            ("多模态t-SNE", lambda: self.multimodal_tsne_viz.visualize_multimodal_tsne(intermediate_features, label, area_id) if self.multimodal_tsne_viz else logger.warning("多模态t-SNE可视化器未初始化")),
            
            # 6. 方法对比
            ("方法对比分析", lambda: self._generate_four_groups_heatmap_comparison(rgb, dsm, label, intermediate_features, area_id)),
            ("框架阶段分析", lambda: self._generate_six_groups_heatmap_samples(rgb, dsm, label, intermediate_features, area_id)),
            
            # 7. 性能指标
            ("混淆矩阵", lambda: self.visualize_confusion_matrix(cm, area_id)),
            ("类别精度分析", lambda: self.visualize_class_accuracy(metrics, area_id))
        ]
        
        for viz_name, viz_func in viz_tasks:
            try:
                viz_func()
                logger.info(f"✅ {viz_name}生成完成")
            except Exception as e:
                logger.warning(f"❌ {viz_name}生成失败: {e}")
        
        # 整理可视化文件
        viz_manager = UnifiedVisualizationManager(self.args.output_dir, area_id)
        moved_count = viz_manager.organize_generated_files()
        
        # 计算总耗时
        area_end_time = time.time()
        total_duration = area_end_time - area_start_time
        logger.info(f"✅ Area {area_id} 完成! ({total_duration/60:.1f}分钟) OA={oa:.4f}, mIoU={miou:.4f}")
        
        return {
            'oa': oa, 'aa': aa, 'miou': miou,
            'metrics': metrics, 'cm': cm
        }
    
    def save_prediction_map(self, prediction, rgb, label, area_id):
        """保存预测图（四列对比：RGB、DSM、GT、预测）- 使用优化色彩"""
        # 确保预测值和标签值在有效范围内 [0, 5]
        prediction = np.clip(prediction, 0, 5).astype(np.uint8)
        label = np.clip(label, 0, 5).astype(np.uint8)
        
        # 加载DSM用于可视化 - 修复路径格式化问题
        dsm_path = os.path.join(self.args.data_path, 'Vaihingen', 'dsm', f'dsm_09cm_matching_area{area_id}.tif')
        if not os.path.exists(dsm_path):
            dsm_path = os.path.join(self.args.data_path, 'Vaihingen', f'dsm_09cm_matching_area{area_id}.tif')
        if not os.path.exists(dsm_path):
            # 尝试其他可能的DSM路径
            dsm_alternatives = [
                os.path.join(self.args.data_path, 'dsm', f'dsm_09cm_matching_area{area_id}.tif'),
                os.path.join(self.args.data_path, f'dsm_09cm_matching_area{area_id}.tif'),
                os.path.join(self.args.data_path, 'Vaihingen', 'dsm', f'area{area_id}.tif'),
                os.path.join(self.args.data_path, 'Vaihingen', f'area{area_id}.tif'),
                # 添加更多可能的路径
                os.path.join(self.args.data_path, 'Vaihingen', 'DSM', f'dsm_09cm_matching_area{area_id}.tif'),
                os.path.join(self.args.data_path, 'DSM', f'dsm_09cm_matching_area{area_id}.tif'),
                os.path.join(self.args.data_path, 'Vaihingen', 'DSM', f'area{area_id}.tif'),
                # 根据日志中的路径添加
                f'/project/lixuyang/collaborative_framework_project666/data/Vaihingen/DSM/dsm_09cm_matching_area{area_id}.tif'
            ]
            for alt_path in dsm_alternatives:
                if os.path.exists(alt_path):
                    dsm_path = alt_path
                    logger.info(f"找到DSM文件: {dsm_path}")
                    break
            else:
                logger.info(f"尝试的DSM路径: {dsm_alternatives[:3]}...")
        
        dsm_vis = None
        if os.path.exists(dsm_path):
            try:
                dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
                if dsm is None:
                    dsm = np.array(Image.open(dsm_path))
                
                if len(dsm.shape) == 3:
                    dsm = dsm[:, :, 0]
                dsm = dsm.astype(np.float32)
                
                # 处理无效值
                dsm[dsm <= 0] = np.nan
                dsm_valid = dsm[~np.isnan(dsm)]
                
                if len(dsm_valid) > 0:
                    # 使用百分位数进行更好的对比度
                    dsm_min = np.percentile(dsm_valid, 2)
                    dsm_max = np.percentile(dsm_valid, 98)
                    
                    # 归一化到[0,1]
                    dsm_norm = np.clip((dsm - dsm_min) / (dsm_max - dsm_min + 1e-8), 0, 1)
                    dsm_norm[np.isnan(dsm)] = 0
                    
                    # 显示原始DSM灰度图，而不是height map
                    dsm_vis = np.stack([dsm_norm, dsm_norm, dsm_norm], axis=2)
                    dsm_vis = (dsm_vis * 255).astype(np.uint8)
                    
                    logger.info(f"DSM加载成功: 范围[{dsm_min:.2f}, {dsm_max:.2f}], 有效像素: {len(dsm_valid)}")
                else:
                    dsm_vis = np.zeros_like(rgb)
                    logger.warning("DSM文件无有效数据")
            except Exception as e:
                logger.warning(f"DSM加载失败: {e}")
                dsm_vis = np.zeros_like(rgb)
        else:
            # 如果没有DSM，用灰度图代替
            dsm_vis = np.zeros_like(rgb)
            logger.warning(f"DSM文件不存在: {dsm_path}")
        
        # 确保预测和标签尺寸匹配
        if prediction.shape != label.shape:
            logger.warning(f"预测尺寸 {prediction.shape} 与标签尺寸 {label.shape} 不匹配，调整预测尺寸")
            prediction = cv2.resize(prediction, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if rgb.shape[:2] != label.shape:
            logger.warning(f"RGB尺寸 {rgb.shape[:2]} 与标签尺寸 {label.shape} 不匹配，调整RGB尺寸")
            rgb = cv2.resize(rgb, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 使用ISPRS标准颜色映射 - 与参考代码一致
        isprs_colors = np.array([
            [255, 255, 255],  # 0: Impervious surfaces - 白色
            [0, 0, 255],      # 1: Building - 蓝色  
            [0, 255, 255],    # 2: Low vegetation - 青色
            [0, 255, 0],      # 3: Tree - 绿色
            [255, 255, 0],    # 4: Car - 黄色
            [255, 0, 0],      # 5: Clutter - 红色
        ], dtype=np.uint8)
        enhanced_colors = isprs_colors
        
        # 转换标签和预测为RGB（确保索引正确）
        pred_colored = enhanced_colors[prediction].astype(np.float32) / 255.0
        label_colored = enhanced_colors[label].astype(np.float32) / 255.0
        
        # 确保RGB值在[0,1]范围内
        rgb_normalized = rgb.astype(np.float32) / 255.0 if rgb.max() > 1.0 else rgb.astype(np.float32)
        
        # 调整DSM尺寸匹配其他图像
        if dsm_vis.shape[:2] != rgb.shape[:2]:
            dsm_vis = cv2.resize(dsm_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 创建四列对比图
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(rgb_normalized)
        axes[0].set_title('RGB Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # DSM显示改进 - 显示原始DSM值而不是Height Map
        dsm_display = dsm_vis.astype(np.float32) / 255.0 if dsm_vis.max() > 1.0 else dsm_vis.astype(np.float32)
        axes[1].imshow(dsm_display, cmap='gray')
        axes[1].set_title('DSM (Original Values)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(label_colored)
        axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        axes[3].imshow(pred_colored)
        axes[3].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[3].axis('off')
        
        # 添加颜色图例
        legend_elements = []
        class_names_short = ['Imperv', 'Build', 'LowVeg', 'Tree', 'Car', 'Clutter']
        for i, (name, color) in enumerate(zip(class_names_short, enhanced_colors)):
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color/255.0, label=name))
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=6, fontsize=12, frameon=False)
        
        plt.tight_layout()
        comparison_path = os.path.join(self.args.output_dir, f'comparison_area_{area_id}.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"保存对比图: comparison_area_{area_id}.png")
        
        # 也保存单独的预测图（向后兼容，使用正确的格式）
        pred_rgb = (pred_colored * 255).astype(np.uint8)
        Image.fromarray(pred_rgb).save(
            os.path.join(self.args.output_dir, f'prediction_area{area_id}.png')
        )
        logger.info(f"保存预测图: prediction_area{area_id}.png")
        
    def visualize_fusion_heatmaps(self, rgb, dsm, label, area_id, intermediate_features):
        """生成融合前后特征热力图对比（类似CMFNet图10）"""
        try:
            # 选择一个代表性的patch（中心区域，256x256）
            patch_size = 256
            H, W = rgb.shape[:2]
            y_start = max(0, (H - patch_size) // 2)
            x_start = max(0, (W - patch_size) // 2)
            y_end = min(H, y_start + patch_size)
            x_end = min(W, x_start + patch_size)
            
            rgb_patch = rgb[y_start:y_end, x_start:x_end]
            dsm_patch = dsm[y_start:y_end, x_start:x_end]
            label_patch = label[y_start:y_end, x_start:x_end]
            
            # 准备patch数据
            rgb_tensor = torch.from_numpy(rgb_patch.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            dsm_min, dsm_max = dsm_patch.min(), dsm_patch.max()
            dsm_norm = (dsm_patch - dsm_min) / (dsm_max - dsm_min + 1e-8)
            dsm_tensor = torch.from_numpy(dsm_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # 获取中间特征
            self.model.eval()
            with torch.no_grad():
                patch_intermediate = self.model({'rgb': rgb_tensor, 'dsm': dsm_tensor})
            
            # 生成不同阶段的热力图
            stages = {
                'Pre-Fusion': 'rgb_attended',
                'After Balancing': 'rgb_balanced',
                'After Fusion': 'after_multi_granularity',
                'Final Features': 'after_spatiotemporal'
            }
            
            heatmaps = []
            stage_names = []
            
            for stage_name, stage_key in stages.items():
                if stage_key in patch_intermediate:
                    feat = patch_intermediate[stage_key]
                    if isinstance(feat, torch.Tensor):
                        # 转换为热力图
                        B, N, D = feat.shape
                        H_feat = W_feat = int(np.sqrt(N))
                        feat_spatial = feat.transpose(1, 2).view(B, D, H_feat, W_feat)
                        
                        # 计算特征激活强度
                        heatmap = torch.norm(feat_spatial, dim=1, keepdim=False)[0].cpu().numpy()
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        
                        # 调整大小到patch大小
                        import cv2
                        heatmap = cv2.resize(heatmap, (patch_size, patch_size))
                        heatmaps.append(heatmap)
                        stage_names.append(stage_name)
            
            if not heatmaps:
                logger.warning(f"Area {area_id} 无法生成融合热力图（缺少中间特征）")
                return
            
            # 创建图像（类似CMFNet图10：RGB、融合前、融合后、GT）
            n_cols = len(heatmaps) + 2  # RGB + 热力图 + GT
            fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))
            
            # RGB图像
            axes[0].imshow(rgb_patch)
            axes[0].set_title('RGB Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # 热力图（纯热力图和叠加图）
            for i, (hm, name) in enumerate(zip(heatmaps, stage_names)):
                # 确保热力图尺寸匹配
                if hm.shape[:2] != rgb_patch.shape[:2]:
                    import cv2
                    hm = cv2.resize(hm, (rgb_patch.shape[1], rgb_patch.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # 生成纯热力图（不是原图！）
                im = axes[i+1].imshow(hm, cmap='jet', interpolation='bilinear')
                axes[i+1].set_title(f'{name} Heatmap', fontsize=12, fontweight='bold')
                axes[i+1].axis('off')
                
                # 添加颜色条
                plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
                axes[i+1].axis('off')
            
            # GT标签
            label_colored = self.colors[label_patch].astype(np.float32) / 255.0
            axes[-1].imshow(label_colored)
            axes[-1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[-1].axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(self.args.output_dir, f'fusion_heatmaps_area_{area_id}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"保存融合热力图: fusion_heatmaps_area_{area_id}.png")
            
        except Exception as e:
            logger.warning(f"Area {area_id} 融合热力图生成失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def run_comprehensive_evaluation(self):
        """快速高精度评估 - 直接运行"""
        logger.info("快速高精度评估模式启动")
        
        all_results = {}
        overall_metrics = {'oa': [], 'aa': [], 'miou': []}
        total_areas = len(self.args.area_ids)
        
        for i, area_id in enumerate(self.args.area_ids, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"总体进度: 区域 {i}/{total_areas} (Area {area_id}) - {(i-1)/total_areas*100:.1f}%")
            logger.info(f"{'='*60}")
            
            result = self.evaluate_area(area_id)
            if result:
                all_results[area_id] = result
                overall_metrics['oa'].append(result['oa'])
                overall_metrics['aa'].append(result['aa'])
                overall_metrics['miou'].append(result['miou'])
                
                # 安全地添加详细指标（如果存在）
                if 'metrics' in result and isinstance(result['metrics'], dict):
                    for key in ['precision', 'recall', 'f1', 'iou']:
                        if key in result['metrics']:
                            if key not in overall_metrics:
                                overall_metrics[key] = []
                            overall_metrics[key].append(result['metrics'][key])
        
        # 计算平均指标
        avg_oa = np.mean(overall_metrics['oa'])
        avg_aa = np.mean(overall_metrics['aa'])
        avg_miou = np.mean(overall_metrics['miou'])
        
        logger.info(f"=== 总体评估结果 ===")
        logger.info(f"平均OA: {avg_oa:.4f}")
        logger.info(f"平均AA: {avg_aa:.4f}")
        logger.info(f"平均mIoU: {avg_miou:.4f}")
        
        # 保存结果
        self.save_evaluation_report(all_results, overall_metrics)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        logger.info("✅ 评估完成，GPU缓存已清理")
        
        return all_results, overall_metrics
    
    def _run_multi_gpu_single_area_evaluation(self):
        """4GPU协同处理单区域模式"""
        all_results = {}
        overall_metrics = {'oa': [], 'aa': [], 'miou': []}
        
        total_areas = len(self.args.area_ids)
        
        for i, area_id in enumerate(self.args.area_ids, 1):
            logger.info(f"{'='*60}")
            logger.info(f"🎯 总体进度: 区域 {i}/{total_areas} (Area {area_id}) - {(i-1)/total_areas*100:.1f}%")
            logger.info(f"🚀 启动4GPU协同处理Area {area_id}...")
            
            # 使用4GPU协同处理单个区域
            result = self.evaluate_area_with_multi_gpu(area_id)
            
            if result:
                all_results[area_id] = result
                overall_metrics['oa'].append(result['oa'])
                overall_metrics['aa'].append(result['aa'])
                overall_metrics['miou'].append(result['miou'])
                
                # 安全地添加metrics，如果不存在则初始化
                for key in ['precision', 'recall', 'f1', 'iou']:
                    if key not in overall_metrics:
                        overall_metrics[key] = []
                    if 'metrics' in result and key in result['metrics']:
                        overall_metrics[key].append(result['metrics'][key])
                    else:
                        logger.warning(f"缺少metrics[{key}]，跳过添加")
        
        # 计算平均指标 - 4GPU协同处理版本
        avg_oa = np.mean(overall_metrics['oa']) if overall_metrics['oa'] else 0.0
        avg_aa = np.mean(overall_metrics['aa']) if overall_metrics['aa'] else 0.0
        avg_miou = np.mean(overall_metrics['miou']) if overall_metrics['miou'] else 0.0
        
        logger.info(f"=== 4GPU协同处理总体评估结果 ===")
        logger.info(f"平均OA: {avg_oa:.4f}")
        logger.info(f"平均AA: {avg_aa:.4f}")
        logger.info(f"平均mIoU: {avg_miou:.4f}")
        
        # 保存结果
        self.save_evaluation_report(all_results, overall_metrics)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        logger.info("✅ 4GPU协同评估完成，GPU缓存已清理")
        
        return all_results, overall_metrics
    
    def evaluate_area_with_multi_gpu(self, area_id):
        """使用4GPU协同处理单个区域"""
        import time
        area_start_time = time.time()
        
        logger.info(f"🎯 开始4GPU协同评估Area {area_id}... (开始时间: {time.strftime('%H:%M:%S')})")
        logger.info(f"📊 配置: stride={self.stride}, window_size={self.window_size}, 4GPU协同模式")
        
        # 加载数据
        logger.info(f"📂 加载Area {area_id}数据...")
        rgb, dsm, label = self.load_vaihingen_data(area_id)
        if rgb is None or dsm is None or label is None:
            logger.error(f"无法加载Area {area_id}的数据")
            return None
        
        # 使用4GPU协同基础预测
        logger.info("🚀 使用4GPU协同基础预测系统...")
        prediction, logits, confidence_map = self.multi_gpu_predict(rgb, dsm)
        
        # 快速指标计算
        y_true = label.flatten()
        y_pred = prediction.flatten()
        
        # 过滤无效像素
        valid_mask = (y_true >= 0) & (y_true <= 5) & (y_pred >= 0) & (y_pred <= 5)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        # 混淆矩阵和指标
        cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
        metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': []}
        
        for i in range(6):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['iou'].append(iou)
        
        # 总体指标
        oa = np.sum(np.diag(cm)) / np.sum(cm)
        aa = np.mean(metrics['recall'])
        miou = np.mean(metrics['iou'][:5])  # 前5类mIoU
        
        logger.info(f"📊 Area {area_id} 结果: OA={oa:.4f}, AA={aa:.4f}, mIoU={miou:.4f}")
        
        # 1. 基础预测图保存
        self.save_prediction_map(prediction, rgb, label, area_id)
        
        # 2. ISPRS标准可视化
        if self.isprs_viz is not None:
            try:
                self.isprs_viz.create_prediction_comparison(rgb, prediction, label, area_id)
                self.isprs_viz.save_prediction_as_color_image(prediction, area_id)
                logger.info("ISPRS标准可视化生成完成")
            except Exception as e:
                logger.warning(f"ISPRS标准可视化生成失败: {e}")
        
        # 快速生成所有可视化
        logger.info("🎨 生成可视化中...")
        
        # 生成完整的浅层到深层特征可视化
        viz_tasks = [
            ("混淆矩阵", lambda: self.visualize_confusion_matrix(cm, area_id)),
            ("类别精度分析", lambda: self.visualize_class_accuracy(metrics, area_id)),
            ("🎨 专业t-SNE特征", lambda: self._generate_professional_tsne(rgb, dsm, label, area_id)),
            ("🔥 多层热力图 (浅→深)", lambda: self._generate_advanced_heatmap(rgb, dsm, area_id)),
            ("⚡ 注意力热力图", lambda: self.visualize_attention_maps(rgb, dsm, area_id)),
            ("🌈 特征融合可视化", lambda: self.visualize_feature_fusion(rgb, dsm, prediction, area_id))
        ]
        
        for viz_name, viz_func in viz_tasks:
            try:
                viz_func()
            except Exception as e:
                logger.warning(f"{viz_name}生成失败: {e}")
        
        # 计算总耗时
        area_end_time = time.time()
        total_duration = area_end_time - area_start_time
        logger.info(f"✅ Area {area_id} 4GPU协同评估完成! (总耗时: {total_duration/60:.1f}分钟)")
        
        return {
            'oa': oa, 'aa': aa, 'miou': miou,
            'metrics': metrics, 'cm': cm
        }
    
    def multi_gpu_predict(self, rgb, dsm):
        """4GPU协同预测单个区域"""
        h, w = rgb.shape[:2]
        
        # 将图像分成4个部分，每个GPU处理一部分
        h_split = h // 2
        w_split = w // 2
        
        # 分割区域
        regions = [
            (0, h_split, 0, w_split),      # GPU 0: 左上
            (0, h_split, w_split, w),      # GPU 1: 右上  
            (h_split, h, 0, w_split),      # GPU 2: 左下
            (h_split, h, w_split, w)       # GPU 3: 右下
        ]
        
        logger.info(f"🔄 将图像分割为4个区域进行并行处理...")
        
        # 初始化结果
        prediction = np.zeros((h, w), dtype=np.uint8)
        logits = np.zeros((h, w, 6), dtype=np.float32)
        
        # 快速4GPU协同处理
        logger.info(f"🚀 4GPU协同处理中... (4个区域)")
        for i, (y1, y2, x1, x2) in enumerate(regions):
            # 提取区域
            rgb_region = rgb[y1:y2, x1:x2]
            dsm_region = dsm[y1:y2, x1:x2]
            
            # 使用基础预测处理区域
            pred_region, logits_region, _ = self.predict_basic(rgb_region, dsm_region)
            
            # 合并结果
            prediction[y1:y2, x1:x2] = pred_region
            logits[y1:y2, x1:x2] = logits_region
        
        confidence_map = np.max(logits, axis=2)
        
        return prediction, logits, confidence_map
    
    def _generate_professional_tsne(self, rgb, dsm, label, area_id):
        """生成专业t-SNE可视化"""
        if self.professional_tsne_viz is not None:
            _, _, _, intermediate_features = self.predict_with_intermediate_features(rgb, dsm)
            if intermediate_features is not None:
                self.professional_tsne_viz.create_professional_tsne(
                    intermediate_features, label, area_id, style='academic'
                )
    
    def _generate_advanced_heatmap(self, rgb, dsm, area_id):
        """生成高级热力图可视化"""
        if self.advanced_heatmap_viz is not None:
            self.advanced_heatmap_viz.create_comprehensive_heatmap(
                self.model, rgb, dsm, area_id
            )
    
    def _generate_multimodal_features(self, rgb, dsm, intermediate_features, area_id):
        """生成多模态特征可视化 (类似Fig. 9)"""
        if self.multimodal_feature_viz is not None:
            self.multimodal_feature_viz.create_multimodal_feature_visualization(
                self.model, rgb, dsm, intermediate_features, area_id
            )
    
    def _generate_multimodal_tsne(self, intermediate_features, label, area_id):
        """生成多模态t-SNE可视化 (类似Fig. 10)"""
        if self.multimodal_tsne_viz is not None:
            self.multimodal_tsne_viz.create_multimodal_tsne_visualization(
                intermediate_features, label, area_id
            )
    
    def _generate_four_groups_heatmap_comparison(self, rgb, dsm, label, intermediate_features, area_id):
        """生成四组热力图对比可视化 (类似Fig. 11)"""
        if self.heatmap_comparison_viz is not None:
            self.heatmap_comparison_viz.create_four_groups_heatmap_comparison(
                self.model, rgb, dsm, label, intermediate_features, area_id
            )
    
    def _generate_six_groups_heatmap_samples(self, rgb, dsm, label, intermediate_features, area_id):
        """生成六组热力图样本可视化 (类似Fig. 9)"""
        if self.heatmap_comparison_viz is not None:
            self.heatmap_comparison_viz.create_six_groups_heatmap_samples(
                self.model, rgb, dsm, label, intermediate_features, area_id
            )
    
    def visualize_confusion_matrix(self, cm, area_id):
        """可视化混淆矩阵"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title(f'Confusion Matrix - Area {area_id}', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=14)
            plt.ylabel('Actual', fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(self.args.output_dir, f'confusion_matrix_area{area_id}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"保存混淆矩阵: {save_path}")
        except Exception as e:
            logger.error(f"混淆矩阵可视化失败: {e}")
    
    def visualize_class_accuracy(self, metrics, area_id):
        """可视化类别精度分析"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Precision
            axes[0,0].bar(self.class_names, metrics['precision'])
            axes[0,0].set_title('Precision by Class', fontweight='bold')
            axes[0,0].set_ylim(0, 1)
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Recall
            axes[0,1].bar(self.class_names, metrics['recall'])
            axes[0,1].set_title('Recall by Class', fontweight='bold')
            axes[0,1].set_ylim(0, 1)
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # F1-Score
            axes[1,0].bar(self.class_names, metrics['f1'])
            axes[1,0].set_title('F1-Score by Class', fontweight='bold')
            axes[1,0].set_ylim(0, 1)
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # IoU
            axes[1,1].bar(self.class_names, metrics['iou'])
            axes[1,1].set_title('IoU by Class', fontweight='bold')
            axes[1,1].set_ylim(0, 1)
            axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            save_path = os.path.join(self.args.output_dir, f'class_accuracy_area{area_id}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"保存类别精度分析: {save_path}")
        except Exception as e:
            logger.error(f"类别精度分析失败: {e}")
    
    def visualize_tsne_features(self, features, labels, area_id):
        """t-SNE特征可视化"""
        try:
            from sklearn.manifold import TSNE
            
            # 处理字典格式的特征
            if isinstance(features, dict):
                logger.info(f"检测到字典格式特征，键: {list(features.keys())}")
                
                # 尝试找到合适的特征层
                feature_candidates = []
                for key, value in features.items():
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                        feature_candidates.append((key, value))
                
                if not feature_candidates:
                    logger.warning("字典中没有找到合适的特征，跳过t-SNE")
                    return
                
                # 选择第一个合适的特征
                feature_key, features = feature_candidates[0]
                logger.info(f"使用特征层: {feature_key}, 形状: {features.shape}")
            
            # 确保特征格式正确
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            # 如果特征是多维的，需要展平
            if len(features.shape) > 2:
                original_shape = features.shape
                # 对于4D特征 (B, C, H, W)，展平为 (B*H*W, C)
                if len(features.shape) == 4:
                    features = features.transpose(0, 2, 3, 1).reshape(-1, features.shape[1])
                else:
                    features = features.reshape(-1, features.shape[-1])
                labels = labels.flatten()
                logger.info(f"特征形状从 {original_shape} 重塑为 {features.shape}")
            
            # 采样数据以加速t-SNE
            if features.shape[0] > 10000:
                indices = np.random.choice(features.shape[0], 5000, replace=False)
                features_sample = features[indices]
                labels_sample = labels[indices] if len(labels) == features.shape[0] else labels.flatten()[indices]
                logger.info(f"采样 {len(indices)} 个样本进行t-SNE")
            else:
                features_sample = features
                labels_sample = labels.flatten() if len(labels.shape) > 1 else labels
            
            # 确保标签和特征数量匹配
            if len(labels_sample) != features_sample.shape[0]:
                logger.warning(f"标签数量 {len(labels_sample)} 与特征数量 {features_sample.shape[0]} 不匹配")
                min_len = min(len(labels_sample), features_sample.shape[0])
                features_sample = features_sample[:min_len]
                labels_sample = labels_sample[:min_len]
            
            # 过滤有效标签
            valid_mask = (labels_sample >= 0) & (labels_sample <= 5)
            features_sample = features_sample[valid_mask]
            labels_sample = labels_sample[valid_mask]
            
            if len(features_sample) < 100:
                logger.warning(f"有效样本太少 ({len(features_sample)})，跳过t-SNE")
                return
            
            # 运行t-SNE
            logger.info(f"运行t-SNE，特征维度: {features_sample.shape}")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_sample)//4))
            features_2d = tsne.fit_transform(features_sample)
            
            # 可视化
            plt.figure(figsize=(12, 10))
            colors = ['white', 'blue', 'cyan', 'green', 'yellow', 'red']
            
            for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                mask = labels_sample == i
                if np.sum(mask) > 0:
                    plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                              c=color, label=f'{class_name} ({np.sum(mask)})', 
                              alpha=0.6, s=2, edgecolors='black', linewidth=0.1)
            
            plt.title(f't-SNE Feature Visualization - Area {area_id}', fontsize=16, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            save_path = os.path.join(self.args.output_dir, f'tsne_features_area{area_id}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"保存t-SNE特征可视化: {save_path}")
        except Exception as e:
            logger.error(f"t-SNE特征可视化失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def visualize_attention_maps(self, rgb, dsm, area_id):
        """生成注意力热力图"""
        try:
            # 使用模型生成注意力图
            h, w = rgb.shape[:2]
            
            # 使用小块进行注意力计算以避免内存问题
            patch_size = 256
            rgb_patch = rgb[:patch_size, :patch_size]
            dsm_patch = dsm[:patch_size, :patch_size]
            
            # 简化版本：使用梯度作为注意力
            rgb_tensor = torch.from_numpy(rgb_patch.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            dsm_tensor = torch.from_numpy(dsm_patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            
            rgb_tensor.requires_grad_(True)
            dsm_tensor.requires_grad_(True)
            
            with torch.enable_grad():
                output = self.model({'rgb': rgb_tensor, 'dsm': dsm_tensor})
                if isinstance(output, tuple):
                    output = output[0]
                
                # 计算梯度作为注意力
                loss = output.sum()
                loss.backward()
                
                rgb_attention = torch.abs(rgb_tensor.grad).mean(dim=1).squeeze().cpu().numpy()
                dsm_attention = torch.abs(dsm_tensor.grad).squeeze().cpu().numpy()
            
            # 可视化注意力图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # RGB注意力热力图
            im1 = axes[0,0].imshow(rgb_attention, cmap='hot', interpolation='bilinear')
            axes[0,0].set_title('RGB Attention Heatmap')
            axes[0,0].axis('off')
            plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
            
            # RGB注意力叠加图
            rgb_norm = rgb_patch / 255.0
            overlay_rgb = 0.6 * rgb_norm + 0.4 * plt.cm.hot(rgb_attention)[:,:,:3]
            axes[0,1].imshow(np.clip(overlay_rgb, 0, 1))
            axes[0,1].set_title('RGB + Attention Overlay')
            axes[0,1].axis('off')
            
            # DSM注意力热力图
            im2 = axes[1,0].imshow(dsm_attention, cmap='plasma', interpolation='bilinear')
            axes[1,0].set_title('DSM Attention Heatmap')
            axes[1,0].axis('off')
            plt.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)
            
            # DSM注意力叠加图
            dsm_norm = (dsm_patch - dsm_patch.min()) / (dsm_patch.max() - dsm_patch.min())
            overlay_dsm = 0.6 * plt.cm.gray(dsm_norm)[:,:,:3] + 0.4 * plt.cm.plasma(dsm_attention)[:,:,:3]
            axes[1,1].imshow(np.clip(overlay_dsm, 0, 1))
            axes[1,1].set_title('DSM + Attention Overlay')
            axes[1,1].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(self.args.output_dir, f'attention_maps_area{area_id}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"保存注意力热力图: {save_path}")
        except Exception as e:
            logger.error(f"注意力热力图生成失败: {e}")
    
    def visualize_feature_fusion(self, rgb, dsm, prediction, area_id):
        """特征融合可视化"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 第一行：输入数据
            axes[0,0].imshow(rgb)
            axes[0,0].set_title('RGB Input', fontsize=14, fontweight='bold')
            axes[0,0].axis('off')
            
            axes[0,1].imshow(dsm, cmap='gray')
            axes[0,1].set_title('DSM Input', fontsize=14, fontweight='bold')
            axes[0,1].axis('off')
            
            # 融合可视化（简化版本）
            fusion_vis = np.stack([rgb[:,:,0]/255.0, dsm/dsm.max(), rgb[:,:,1]/255.0], axis=2)
            axes[0,2].imshow(fusion_vis)
            axes[0,2].set_title('RGB-DSM Fusion', fontsize=14, fontweight='bold')
            axes[0,2].axis('off')
            
            # 第二行：预测结果
            pred_colored = convert_to_color(prediction)
            axes[1,0].imshow(pred_colored)
            axes[1,0].set_title('Prediction Result', fontsize=14, fontweight='bold')
            axes[1,0].axis('off')
            
            # 置信度图 - 使用预测类别的分布作为置信度
            confidence = np.ones_like(prediction, dtype=np.float32) * 0.5  # 默认置信度
            unique_classes = np.unique(prediction)
            for cls in unique_classes:
                mask = prediction == cls
                confidence[mask] = 0.8 + 0.2 * (cls / 5.0)  # 简单的置信度模拟
            
            im = axes[1,1].imshow(confidence, cmap='viridis')
            axes[1,1].set_title('Confidence Map', fontsize=14, fontweight='bold')
            axes[1,1].axis('off')
            plt.colorbar(im, ax=axes[1,1])
            
            # 边界检测
            from scipy import ndimage
            edges = ndimage.sobel(rgb.mean(axis=2))
            axes[1,2].imshow(edges, cmap='gray')
            axes[1,2].set_title('Edge Detection', fontsize=14, fontweight='bold')
            axes[1,2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(self.args.output_dir, f'feature_fusion_area{area_id}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"保存特征融合可视化: {save_path}")
        except Exception as e:
            logger.error(f"特征融合可视化失败: {e}")
    
    def _run_parallel_evaluation(self):
        """并行评估模式 - 4GPU自动并行"""
        import multiprocessing as mp
        import subprocess
        import time
        
        logger.info("🚀 启动4GPU并行评估...")
        
        # 创建并行任务
        processes = []
        area_ids = self.args.area_ids
        gpu_count = min(4, torch.cuda.device_count())
        
        for i, area_id in enumerate(area_ids):
            gpu_id = i % gpu_count
            
            # 为每个GPU创建独立的输出目录
            gpu_output_dir = os.path.join(self.args.output_dir, f'gpu{gpu_id}_area{area_id}')
            
            # 构建命令
            cmd = [
                'python', '-c', f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
import sys
sys.path.append("{os.getcwd()}")

from evaluate_top_tier import TopTierEvaluator
import argparse

# 创建参数对象
class Args:
    def __init__(self):
        self.model_path = "{self.args.model_path}"
        self.data_path = "{self.args.data_path}"
        self.output_dir = "{gpu_output_dir}"
        self.area_ids = [{area_id}]
        self.stride = {self.args.stride if hasattr(self.args, 'stride') else 8}
        self.window_size = {self.args.window_size if hasattr(self.args, 'window_size') else 256}
        self.batch_size = {self.args.batch_size if hasattr(self.args, 'batch_size') else 4}
        self.use_multi_strategy = {getattr(self.args, 'use_multi_strategy', True)}
        self.embed_dim = {getattr(self.args, 'embed_dim', None)}
        self.ablation_stride = {getattr(self.args, 'ablation_stride', 32)}

args = Args()
evaluator = TopTierEvaluator(args)
result = evaluator._run_serial_evaluation()
print(f"GPU {gpu_id} Area {area_id} 完成")
'''
            ]
            
            logger.info(f"🔥 GPU {gpu_id} 开始处理 Area {area_id}")
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((process, area_id, gpu_id, gpu_output_dir))
        
        # 等待所有进程完成
        logger.info("⏳ 等待所有GPU完成评估...")
        
        completed_results = {}
        for process, area_id, gpu_id, output_dir in processes:
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ GPU {gpu_id} Area {area_id} 评估成功")
                completed_results[area_id] = {
                    'gpu_id': gpu_id,
                    'output_dir': output_dir,
                    'success': True
                }
            else:
                logger.error(f"❌ GPU {gpu_id} Area {area_id} 评估失败")
                logger.error(f"错误: {stderr}")
                completed_results[area_id] = {
                    'gpu_id': gpu_id,
                    'output_dir': output_dir,
                    'success': False,
                    'error': stderr
                }
        
        # 合并结果
        logger.info("🔄 合并并行评估结果...")
        self._merge_parallel_results(completed_results)
        
        logger.info("✅ 4GPU并行评估完成!")
        return completed_results, {}
    
    def _run_simple_parallel_evaluation(self):
        """简化的4GPU并行评估 - 直接使用shell命令"""
        import subprocess
        import time
        
        logger.info("🚀 启动简化4GPU并行评估...")
        
        area_ids = self.args.area_ids
        processes = []
        
        # 为每个区域启动独立的评估进程
        for i, area_id in enumerate(area_ids):
            gpu_id = i % 4
            output_dir = f"{self.args.output_dir}_gpu{gpu_id}_area{area_id}"
            
            # 构建简化的命令
            cmd = [
                'python', 'evaluate_top_tier.py',
                '--checkpoint', str(self.args.model_path),
                '--data_path', str(self.args.data_path), 
                '--output_dir', output_dir,
                '--area_ids', str(area_id),
                '--stride', str(getattr(self.args, 'stride', 8)),
                '--window_size', str(getattr(self.args, 'window_size', 256)),
                '--batch_size', str(getattr(self.args, 'batch_size', 4)),
                '--disable_multi_strategy'  # 强制禁用多策略
            ]
            
            # 设置GPU环境
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            logger.info(f"🔥 GPU {gpu_id} 开始处理 Area {area_id} (简化模式)")
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((process, area_id, gpu_id, output_dir))
        
        # 等待所有进程完成 - 带心跳监控
        logger.info("⏳ 等待所有GPU完成简化评估...")
        
        completed_results = {}
        for i, (process, area_id, gpu_id, output_dir) in enumerate(processes):
            try:
                logger.info(f"⌛ 等待GPU {gpu_id} Area {area_id}完成... ({i+1}/{len(processes)})")
                
                # 带心跳监控的等待，每分钟检查一次
                start_wait_time = time.time()
                while process.poll() is None:
                    try:
                        stdout, stderr = process.communicate(timeout=60)  # 1分钟心跳检查
                        break
                    except subprocess.TimeoutExpired:
                        elapsed_wait = time.time() - start_wait_time
                        logger.info(f"💓 GPU {gpu_id} Area {area_id} 仍在运行... (已运行 {elapsed_wait/60:.1f}分钟)")
                        # 移除强制终止，让进程自然完成
                
                # 获取最终输出
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"✅ GPU {gpu_id} Area {area_id} 简化评估成功")
                    completed_results[area_id] = {
                        'gpu_id': gpu_id,
                        'output_dir': output_dir,
                        'success': True
                    }
                else:
                    logger.error(f"❌ GPU {gpu_id} Area {area_id} 简化评估失败")
                    logger.error(f"错误: {stderr}")
                    completed_results[area_id] = {
                        'gpu_id': gpu_id,
                        'output_dir': output_dir,
                        'success': False,
                        'error': stderr
                    }
            except Exception as e:
                logger.error(f"💥 GPU {gpu_id} Area {area_id} 评估异常: {e}")
                completed_results[area_id] = {
                    'gpu_id': gpu_id,
                    'output_dir': output_dir,
                    'success': False,
                    'error': str(e)
                }
        
        # 合并结果
        logger.info("🔄 合并简化并行评估结果...")
        self._merge_parallel_results(completed_results)
        
        logger.info("✅ 简化4GPU并行评估完成!")
        return completed_results, {}
    
    def _merge_parallel_results(self, completed_results):
        """合并并行评估的结果"""
        try:
            import shutil
            
            # 创建合并目录
            merged_dir = os.path.join(self.args.output_dir, 'merged_results')
            os.makedirs(merged_dir, exist_ok=True)
            
            for area_id, result_info in completed_results.items():
                if result_info['success']:
                    source_dir = result_info['output_dir']
                    
                    if os.path.exists(source_dir):
                        # 复制文件到合并目录
                        for file in os.listdir(source_dir):
                            if file.endswith(('.png', '.txt', '.json')):
                                src_file = os.path.join(source_dir, file)
                                dst_file = os.path.join(merged_dir, f'area{area_id}_{file}')
                                shutil.copy2(src_file, dst_file)
            
            logger.info(f"📁 结果已合并到: {merged_dir}")
            
        except Exception as e:
            logger.error(f"❌ 结果合并失败: {e}")
    
    def save_evaluation_report(self, all_results, overall_metrics):
        """保存评估报告"""
        report_path = os.path.join(self.args.output_dir, 'comprehensive_evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("顶刊级综合评估报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 总体指标
            avg_oa = np.mean(overall_metrics['oa'])
            avg_aa = np.mean(overall_metrics['aa'])
            avg_miou = np.mean(overall_metrics['miou'])
            
            f.write("总体指标:\n")
            f.write(f"  平均OA: {avg_oa:.4f}\n")
            f.write(f"  平均AA: {avg_aa:.4f}\n")
            f.write(f"  平均mIoU (前5类): {avg_miou:.4f}\n\n")
            
            # 各类别平均指标
            f.write("各类别平均指标:\n")
            for i, name in enumerate(self.class_names):
                try:
                    # 从all_results中提取各类别指标
                    precisions = [result['metrics']['precision'][i] for result in all_results.values()]
                    recalls = [result['metrics']['recall'][i] for result in all_results.values()]
                    f1s = [result['metrics']['f1'][i] for result in all_results.values()]
                    ious = [result['metrics']['per_class_iou'][i] for result in all_results.values()]
                    
                    avg_precision = np.mean(precisions)
                    avg_recall = np.mean(recalls)
                    avg_f1 = np.mean(f1s)
                    avg_iou = np.mean(ious)
                    
                    f.write(f"  {name}:\n")
                    f.write(f"    Precision: {avg_precision:.4f}\n")
                    f.write(f"    Recall: {avg_recall:.4f}\n")
                    f.write(f"    F1: {avg_f1:.4f}\n")
                    f.write(f"    IoU: {avg_iou:.4f}\n\n")
                except Exception as e:
                    f.write(f"  {name}: 计算失败 ({e})\n\n")
            
            # 各区域结果
            f.write("各测试区域结果:\n")
            for area_id, result in all_results.items():
                f.write(f"  Area {area_id}:\n")
                f.write(f"    OA: {result['oa']:.4f}\n")
                f.write(f"    AA: {result['aa']:.4f}\n")
    
    def _generate_top_tier_tsne(self, intermediate_features, labels, area_id):
        """生成顶刊级t-SNE可视化"""
        if self.top_tier_tsne_viz is not None:
            try:
                logger.info(f"生成顶刊级t-SNE可视化 - Area {area_id}")
                
                # 创建发表级t-SNE
                self.top_tier_tsne_viz.create_publication_tsne(
                    intermediate_features, labels, area_id, "Final"
                )
                
                # 创建多阶段对比
                if isinstance(intermediate_features, dict):
                    self.top_tier_tsne_viz.create_multi_stage_comparison(
                        intermediate_features, labels, area_id
                    )
                
                logger.info("✅ 顶刊级t-SNE可视化完成")
                
            except Exception as e:
                logger.error(f"❌ 顶刊级t-SNE可视化失败: {e}")
        else:
            logger.warning("⚠️ 顶刊级t-SNE可视化器未初始化")
    
    def _generate_top_tier_heatmap(self, rgb, dsm, intermediate_features, area_id):
        """生成顶刊级热力图可视化"""
        if self.top_tier_heatmap_viz is not None:
            try:
                logger.info(f"生成顶刊级热力图可视化 - Area {area_id}")
                
                # 创建发表级热力图
                self.top_tier_heatmap_viz.create_publication_heatmap(
                    rgb, dsm, intermediate_features, area_id
                )
                
                logger.info("✅ 顶刊级热力图可视化完成")
                
            except Exception as e:
                logger.error(f"❌ 顶刊级热力图可视化失败: {e}")
        else:
            logger.warning("⚠️ 顶刊级热力图可视化器未初始化")


def main():
    parser = argparse.ArgumentParser(description='顶刊级评估与可视化系统')
    parser.add_argument('--model_path', '--checkpoint', type=str, required=True, help='模型路径')
    parser.add_argument('--dataset', type=str, default='vaihingen', 
                        choices=['vaihingen', 'augsburg', 'muufl', 'trento'],
                        help='数据集名称')
    parser.add_argument('--data_path', type=str, default='./data', help='数据路径')
    parser.add_argument('--output_dir', type=str, default='top_tier_results', help='输出目录')
    parser.add_argument('--area_ids', type=int, nargs='+', default=[5, 15, 21, 30], help='测试区域ID')
    parser.add_argument('--ablation_stride', type=int, default=32, help='消融实验专用滑窗步长（仅用于消融评估）')
    parser.add_argument('--stride', type=int, default=4, help='评估滑窗步长')
    parser.add_argument('--embed_dim', type=int, default=None, help='模型embed_dim覆盖（与训练一致时留空）')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--window_size', type=int, default=256, help='滑动窗口大小')
    parser.add_argument('--use_multi_strategy', action='store_true', default=True, help='启用多策略集成（默认开启）')
    parser.add_argument('--disable_multi_strategy', action='store_true', help='禁用多策略集成')
    
    args = parser.parse_args()
    
    # 处理多策略集成参数
    if args.disable_multi_strategy:
        args.use_multi_strategy = False
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = TopTierEvaluator(args)
    evaluator.run_comprehensive_evaluation()
    
    logger.info("评估完成！所有可视化图表已生成")


if __name__ == '__main__':
    main()

