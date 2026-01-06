#!/usr/bin/env python3
"""
ACF (Adaptive Collaborative Fusion) Network
自适应协作融合网络 - 完整实现

包含5个核心模块：
1. Hierarchical Tokenizer (HT) - 分层标记化
2. Cross-Modal Attention (CMA) - 跨模态注意力
3. Adaptive Modality Balancing (AMB) - 自适应模态平衡
4. Multi-Granularity Consistency Fusion (MGCF) - 多粒度一致性融合
5. Spatial-Channel Adaptive Factor (SCAF) - 空间-通道自适应因子

支持数据集：Vaihingen, Augsburg, MUUFL, Trento
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math


class HierarchicalTokenizer(nn.Module):
    """
    模块1: 分层标记化 (Hierarchical Tokenizer)
    将异构模态数据转换为统一的token表示
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 模态适配嵌入 - 动态选择1D/2D/3D卷积
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 层级位置编码
        self.pos_embed = None  # 动态创建
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入特征
        Returns:
            tokens: (B, N, D) token序列
        """
        B, C, H, W = x.shape
        
        # 模态适配嵌入
        x = self.patch_embed(x)  # (B, D, H', W')
        
        # 动态创建位置编码
        _, _, h, w = x.shape
        if self.pos_embed is None or self.pos_embed.shape[-2:] != (h, w):
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.embed_dim, h, w, device=x.device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 空间-序列转换
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # 层归一化
        return self.norm(x)


class CrossModalAttention(nn.Module):
    """
    模块2: 跨模态注意力 (Cross-Modal Attention)
    实现模态间的深度语义关联
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, N, D) 查询模态
            key: (B, N, D) 键模态
            value: (B, N, D) 值模态
        Returns:
            output: (B, N, D) 增强后的特征
        """
        B, N, D = query.shape
        
        # 投影到Q, K, V
        Q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        
        # 残差连接
        return query + self.alpha * out


class AdaptiveModalityBalancing(nn.Module):
    """
    模块3: 自适应模态平衡 (Adaptive Modality Balancing)
    动态校准各模态的贡献度，解决模态显著性偏倚
    """
    def __init__(self, embed_dim: int, num_modalities: int = 2, tau: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.tau = tau  # 贡献度阈值
        
        # 贡献度检测网络
        self.contribution_detector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 梯度平滑系数（强模态）
        self.lambda_smooth = 0.3
        
        # 特征增强系数（弱模态）
        self.beta_enhance = 0.5
        
    def forward(self, modality_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            modality_features: List of (B, N, D) 各模态特征
        Returns:
            balanced_features: List of (B, N, D) 平衡后的特征
        """
        # 调试输出
        if not hasattr(self, '_amb_debug_printed'):
            print(f"\n[DEBUG] AdaptiveModalityBalancing input:")
            for i, feat in enumerate(modality_features):
                print(f"  modality_features[{i}].shape: {feat.shape}")
            self._amb_debug_printed = True
        
        # 计算各模态的贡献度
        contributions = []
        for idx, feat in enumerate(modality_features):
            # (B, N, D) -> (B, D, N) -> (B, 1)
            feat_transposed = feat.transpose(1, 2)
            contrib = self.contribution_detector(feat_transposed)
            
            # 调试：检查contribution_detector的输出
            if not hasattr(self, '_contrib_detector_debug_printed'):
                print(f"[DEBUG] contribution_detector output for modality {idx}:")
                print(f"  feat_transposed.shape: {feat_transposed.shape}")
                print(f"  contrib.shape: {contrib.shape}")
                print(f"  contrib.dim(): {contrib.dim()}")
                self._contrib_detector_debug_printed = True
            
            contributions.append(contrib)
        
        contributions = torch.stack(contributions, dim=1)  # (B, M, 1)
        contributions = contributions.squeeze(-1)  # (B, M) - 移除最后一维
        
        # 调试：检查stack后的形状
        if not hasattr(self, '_contributions_stack_debug_printed'):
            print(f"[DEBUG] After torch.stack and squeeze:")
            print(f"  contributions.shape: {contributions.shape}")
            print(f"  contributions.dim(): {contributions.dim()}")
            self._contributions_stack_debug_printed = True
        
        # 识别强弱模态
        balanced_features = []
        for i, feat in enumerate(modality_features):
            contrib = contributions[:, i:i+1]  # (B, 1)
            contrib = contrib.unsqueeze(-1)  # (B, 1, 1) 用于广播到 (B, N, D)
            
            # 调试输出 - 在计算之前检查形状
            if not hasattr(self, '_amb_contrib_debug_printed'):
                print(f"[DEBUG] AdaptiveModalityBalancing contrib after unsqueeze:")
                print(f"  contributions.shape: {contributions.shape}")
                print(f"  contributions[:, {i}:{i+1}].shape: {contributions[:, i:i+1].shape}")
                print(f"  contrib.shape after unsqueeze(-1): {contrib.shape}")
                print(f"  contrib.dim(): {contrib.dim()}")
                self._amb_contrib_debug_printed = True
            
            if contrib.mean() > self.tau:
                # 强模态：梯度平滑
                balanced_feat = feat * (1 - self.lambda_smooth * torch.sigmoid(contrib - self.tau))
            else:
                # 弱模态：特征增强
                balanced_feat = feat * (1 + self.beta_enhance * F.relu(self.tau - contrib))
            
            # 调试输出 - 在计算之后检查形状
            if not hasattr(self, '_amb_output_debug_printed'):
                print(f"[DEBUG] AdaptiveModalityBalancing output:")
                print(f"  contrib.shape: {contrib.shape}")
                print(f"  feat.shape: {feat.shape}")
                print(f"  balanced_feat.shape: {balanced_feat.shape}")
                self._amb_output_debug_printed = True
            
            balanced_features.append(balanced_feat)
        
        return balanced_features


class MultiGranularityConsistencyFusion(nn.Module):
    """
    模块4: 多粒度一致性融合 (Multi-Granularity Consistency Fusion)
    构建四个语义粒度的特征表示并动态选择
    支持一致性损失约束
    """
    def __init__(self, embed_dim: int, scales: List[int] = [1, 2, 4, 8], 
                 use_consistency_loss: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.scales = scales
        self.use_consistency_loss = use_consistency_loss
        
        # 各粒度的特征提取器
        self.granularity_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((None, None)) if scale == 1 else 
                nn.AdaptiveAvgPool2d((scale, scale))
            ) for scale in scales
        ])
        
        # 粒度选择器
        self.granularity_selector = nn.Sequential(
            nn.Linear(embed_dim * len(scales), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, len(scales)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, N, D) token序列
            H, W: 原始空间尺寸（用于参考，实际会根据N计算）
        Returns:
            fused: (B, N, D) 多粒度融合特征
            consistency_loss: 一致性损失（训练时返回，推理时为None）
        """
        # 确保输入是3维的
        if x.dim() == 2:
            # 如果是(N, D)，添加batch维度
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            # 如果是(B, D, H, W)，转换为(B, N, D)
            B, D, h, w = x.shape
            x = x.flatten(2).transpose(1, 2)
            H, W = h, w  # 更新H, W为实际的空间尺寸
        
        B, N, D = x.shape
        
        # 根据N计算实际的空间尺寸
        # N = h * w，我们需要找到合适的h和w
        import math
        
        if H * W == N:
            # 如果传入的H和W匹配，直接使用
            h, w = H, W
        else:
            # 否则，尝试找到最佳的h和w
            # 优先尝试正方形
            sqrt_n = int(math.sqrt(N))
            if sqrt_n * sqrt_n == N:
                # 完全平方数
                h = w = sqrt_n
            else:
                # 不是完全平方数，找最接近正方形的因子对
                # 从sqrt(N)开始向下搜索
                h = w = sqrt_n
                for h_try in range(sqrt_n, 0, -1):
                    if N % h_try == 0:
                        h = h_try
                        w = N // h_try
                        break
        
        # 转换为空间格式: (B, N, D) -> (B, D, N) -> (B, D, h, w)
        x_transposed = x.transpose(1, 2)  # (B, D, N)
        
        # 调试输出（只打印一次）
        if not hasattr(self, '_debug_printed'):
            print(f"\n[DEBUG] MultiGranularityFusion:")
            print(f"  Input x.shape: {x.shape} = (B={B}, N={N}, D={D})")
            print(f"  Passed H={H}, W={W}, H*W={H*W}")
            print(f"  Calculated h={h}, w={w}, h*w={h*w}")
            print(f"  After transpose(1,2): {x_transposed.shape}")
            print(f"  Target reshape: ({B}, {D}, {h}, {w})")
            self._debug_printed = True
        
        x_spatial = x_transposed.reshape(B, D, h, w)
        
        # 提取各粒度特征
        multi_grain_feats = []
        for encoder in self.granularity_encoders:
            feat = encoder(x_spatial)
            # 上采样回原始尺寸
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            multi_grain_feats.append(feat)
        
        # 计算一致性损失（如果启用且在训练模式）
        consistency_loss = None
        if self.use_consistency_loss and self.training:
            consistency_loss = self._compute_consistency_loss(multi_grain_feats)
        
        # 拼接所有粒度特征
        concat_feats = torch.cat(multi_grain_feats, dim=1)  # (B, D*scales, h, w)
        
        # 计算粒度选择权重
        global_feat = F.adaptive_avg_pool2d(concat_feats, 1).flatten(1)  # (B, D*scales)
        weights = self.granularity_selector(global_feat)  # (B, scales)
        
        # 加权融合
        fused = sum(w.view(B, 1, 1, 1) * feat 
                   for w, feat in zip(weights.unbind(1), multi_grain_feats))
        
        # 转换回序列格式
        fused = fused.flatten(2).transpose(1, 2)  # (B, N, D)
        
        return fused, consistency_loss
    
    def _compute_consistency_loss(self, multi_grain_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        计算多粒度特征的一致性损失
        
        公式: L_consistency = Σ_{i,j∈{1,2,4,8}} ||GAP(F_i) - GAP(F_j)||_2^2
        
        Args:
            multi_grain_feats: List of (B, D, h, w) 各粒度特征
        Returns:
            consistency_loss: 标量损失值
        """
        # 对每个粒度特征进行全局平均池化
        global_feats = []
        for feat in multi_grain_feats:
            # GAP: (B, D, h, w) -> (B, D, 1, 1) -> (B, D)
            global_feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            global_feats.append(global_feat)
        
        # 计算所有粒度对之间的L2距离平方
        consistency_loss = 0.0
        num_pairs = 0
        
        for i in range(len(global_feats)):
            for j in range(i + 1, len(global_feats)):
                # ||GAP(F_i) - GAP(F_j)||_2^2
                diff = global_feats[i] - global_feats[j]
                loss_ij = torch.mean(diff ** 2)
                consistency_loss += loss_ij
                num_pairs += 1
        
        # 平均所有粒度对的损失
        consistency_loss = consistency_loss / num_pairs
        
        return consistency_loss


class SpatialChannelAdaptiveFactor(nn.Module):
    """
    模块5: 空间-通道自适应因子 (Spatial-Channel Adaptive Factor)
    联合建模空间上下文和通道语义
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 空间自适应因子
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 通道自适应因子
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 16, embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token序列
            H, W: 空间尺寸（用于参考，实际会根据N计算）
        Returns:
            enhanced: (B, N, D) 增强后的特征
        """
        # 确保输入是3维的
        if x.dim() == 2:
            # 如果是(N, D)，添加batch维度
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            # 如果是(B, D, H, W)，转换为(B, N, D)
            B, D, h, w = x.shape
            x = x.flatten(2).transpose(1, 2)
            H, W = h, w  # 更新H, W为实际的空间尺寸
        
        B, N, D = x.shape
        
        # 根据N计算实际的空间尺寸
        import math
        
        if H * W == N:
            h, w = H, W
        else:
            # 尝试找到最佳的h和w
            h = w = int(math.sqrt(N))
            if h * w != N:
                # 从sqrt(N)开始向下搜索最接近的因子对
                for h_try in range(int(math.sqrt(N)), 0, -1):
                    if N % h_try == 0:
                        h = h_try
                        w = N // h_try
                        break
        
        # 转换为空间格式: (B, N, D) -> (B, D, N) -> (B, D, h, w)
        x_spatial = x.transpose(1, 2).reshape(B, D, h, w)
        
        # 计算空间注意力
        max_pool = torch.max(x_spatial, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_spatial, dim=1, keepdim=True)
        spatial_feat = torch.cat([max_pool, avg_pool], dim=1)
        spatial_attn = self.spatial_attention(spatial_feat)  # (B, 1, h, w)
        
        # 计算通道注意力
        channel_attn = self.channel_attention(x_spatial)  # (B, D, 1, 1)
        
        # 空间-通道耦合
        coupled_attn = spatial_attn * channel_attn  # (B, D, h, w)
        
        # 特征增强
        enhanced = x_spatial * (1 + coupled_attn)
        
        # 转换回序列格式
        enhanced = enhanced.flatten(2).transpose(1, 2)  # (B, N, D)
        
        return enhanced
        
        # 计算空间注意力
        max_pool = torch.max(x_spatial, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_spatial, dim=1, keepdim=True)
        spatial_feat = torch.cat([max_pool, avg_pool], dim=1)
        spatial_attn = self.spatial_attention(spatial_feat)  # (B, 1, H, W)
        
        # 计算通道注意力
        channel_attn = self.channel_attention(x_spatial)  # (B, D, 1, 1)
        
        # 空间-通道耦合
        coupled_attn = spatial_attn * channel_attn  # (B, D, H, W)
        
        # 特征增强
        enhanced = x_spatial * (1 + coupled_attn)
        
        # 转换回序列格式
        enhanced = enhanced.flatten(2).transpose(1, 2)  # (B, N, D)
        
        return enhanced


class ACFNetwork(nn.Module):
    """
    完整的ACF网络 - 极致精度版本 (目标mIoU>85%)
    支持多数据集：Vaihingen, Augsburg, MUUFL, Trento
    
    5个核心创新模块：
    1. HT (Hierarchical Tokenizer) - 分层标记化
    2. CMA (Cross-Modal Attention) - 跨模态注意力 (3层堆叠)
    3. AMB (Adaptive Modality Balancing) - 自适应模态平衡
    4. MGCF (Multi-Granularity Consistency Fusion) - 多粒度一致性融合 (5尺度)
    5. SCAF (Spatial-Channel Adaptive Factor) - 空间-通道自适应因子
    """
    def __init__(
        self,
        dataset: str = 'vaihingen',
        num_classes: int = 6,
        embed_dim: int = 768,  # ⭐ 256→768，匹配ViT-Base
        num_heads: int = 12,  # ⭐ 8→12，更多注意力头
        patch_size: int = 16,  # ⭐ 4→16，标准ViT配置
        num_cma_layers: int = 3,  # ⭐ 新增：CMA层数
        use_consistency_loss: bool = True,  # ⭐ 新增：是否使用一致性损失
        consistency_loss_weight: float = 0.1  # ⭐ 新增：一致性损失权重
    ):
        super().__init__()
        self.dataset = dataset.lower()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_cma_layers = num_cma_layers
        self.use_consistency_loss = use_consistency_loss
        self.consistency_loss_weight = consistency_loss_weight
        
        # 根据数据集配置模态
        self.modality_config = self._get_modality_config()
        self.num_modalities = len(self.modality_config)
        
        # 模块1: 分层标记化（每个模态一个）
        self.tokenizers = nn.ModuleDict({
            name: HierarchicalTokenizer(channels, embed_dim, patch_size)
            for name, channels in self.modality_config.items()
        })
        
        # 模块2: 跨模态注意力 (⭐ 堆叠3层以增强交互深度)
        self.cross_modal_attention_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads)
            for _ in range(num_cma_layers)
        ])
        
        # 模块3: 自适应模态平衡
        self.modality_balancing = AdaptiveModalityBalancing(embed_dim, self.num_modalities)
        
        # 模块4: 多粒度一致性融合（支持一致性损失）
        self.multi_granularity_fusion = MultiGranularityConsistencyFusion(
            embed_dim, use_consistency_loss=use_consistency_loss
        )
        
        # 模块5: 空间-通道自适应因子
        self.scaf = SpatialChannelAdaptiveFactor(embed_dim)
        
        # 解码器 - 改进版本：使用GELU + BatchNorm + Logits Scaling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)
        )
        
        # 关键修复：添加可学习的logits缩放参数
        # 初始值设为10.0，让模型学习最优缩放
        self.logits_scale = nn.Parameter(torch.ones(1) * 10.0)
        
        # 重要：初始化分类头，使用更大的权重以产生合理的logits范围
        self._init_classification_head()
        
    def _init_classification_head(self):
        """初始化分类头，确保logits有合理的数值范围"""
        # 获取最后一层（分类头）
        classifier = self.decoder[-1]
        if isinstance(classifier, nn.Conv2d):
            # 使用标准初始化 + 较大的std
            # 配合logits_scale参数，不需要过于激进的初始化
            nn.init.normal_(classifier.weight, mean=0.0, std=0.02)
            if classifier.bias is not None:
                # 为每个类别设置不同的初始偏置，避免某个类别占主导
                # 使用小的随机值
                nn.init.uniform_(classifier.bias, -0.1, 0.1)
        
    def _get_modality_config(self) -> Dict[str, int]:
        """根据数据集返回模态配置"""
        configs = {
            'vaihingen': {'rgb': 3, 'dsm': 1},  # IRRG(3) + DSM(1)
            'augsburg': {'hsi': 180, 'sar': 4, 'dsm': 1},  # HSI(180) + PolSAR(4) + DSM(1)
            'muufl': {'hsi': 64, 'lidar': 2},  # HSI(64) + LiDAR(2)
            'trento': {'hsi': 63, 'lidar': 1}  # HSI(63) + LiDAR(1)
        }
        return configs.get(self.dataset, configs['vaihingen'])
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            inputs: Dict of modality tensors
                - 'rgb' or 'hsi': (B, C, H, W)
                - 'dsm' or 'lidar' or 'sar': (B, C, H, W)
        Returns:
            训练模式（self.training=True）:
                output: (B, num_classes, H, W) 分割结果
                aux_losses: Dict包含辅助损失（如一致性损失）
            推理模式（self.training=False）:
                output: (B, num_classes, H, W) 分割结果
        """
        B = list(inputs.values())[0].shape[0]
        H, W = list(inputs.values())[0].shape[2:]
        
        # 步骤1: 分层标记化 (HT)
        tokens = []
        for modality_name in self.modality_config.keys():
            if modality_name in inputs:
                token = self.tokenizers[modality_name](inputs[modality_name])
                tokens.append(token)
                
                # 调试输出
                if not hasattr(self, '_tokenizer_debug_printed'):
                    print(f"\n[DEBUG] Tokenizer output for '{modality_name}':")
                    print(f"  Input shape: {inputs[modality_name].shape}")
                    print(f"  Token shape: {token.shape}")
        
        if not hasattr(self, '_tokenizer_debug_printed'):
            self._tokenizer_debug_printed = True
        
        # 步骤2: 跨模态注意力 (CMA) - ⭐ 堆叠3层深度交互
        # 对每对模态进行多层交互
        enhanced_tokens = tokens
        for layer_idx, cma_layer in enumerate(self.cross_modal_attention_layers):
            layer_enhanced = []
            for i, token_i in enumerate(enhanced_tokens):
                enhanced = token_i
                for j, token_j in enumerate(enhanced_tokens):
                    if i != j:
                        enhanced = cma_layer(enhanced, token_j, token_j)
                layer_enhanced.append(enhanced)
            enhanced_tokens = layer_enhanced
            
            # 调试输出
            if not hasattr(self, '_cma_debug_printed') and layer_idx == 0:
                print(f"[DEBUG] After CrossModalAttention Layer {layer_idx}:")
                for i, tok in enumerate(enhanced_tokens):
                    print(f"  enhanced_tokens[{i}].shape: {tok.shape}")
        
        if not hasattr(self, '_cma_debug_printed'):
            self._cma_debug_printed = True
        
        # 步骤3: 自适应模态平衡 (AMB)
        balanced_tokens = self.modality_balancing(enhanced_tokens)
        
        # 融合所有模态
        fused_token = sum(balanced_tokens) / len(balanced_tokens)
        
        # 调试输出
        if not hasattr(self, '_acf_debug_printed'):
            print(f"\n[DEBUG] ACFNetwork.forward:")
            print(f"  Number of modalities: {len(balanced_tokens)}")
            for i, tok in enumerate(balanced_tokens):
                print(f"  balanced_tokens[{i}].shape: {tok.shape}")
            print(f"  fused_token.shape: {fused_token.shape}")
            # patch_size从配置中获取
            patch_size = list(self.tokenizers.values())[0].patch_size
            print(f"  Expected: (B={B}, N={(H//patch_size)*(W//patch_size)}, D={self.embed_dim})")
            self._acf_debug_printed = True
        
        # 步骤4: 多粒度一致性融合 (MGCF) - 可能返回一致性损失
        # 根据patch_size动态计算h, w
        patch_size = list(self.tokenizers.values())[0].patch_size
        h, w = H // patch_size, W // patch_size
        multi_grain_feat, consistency_loss = self.multi_granularity_fusion(fused_token, h, w)
        
        # 步骤5: 空间-通道自适应因子 (SCAF)
        enhanced_feat = self.scaf(multi_grain_feat, h, w)
        
        # 解码
        feat_spatial = enhanced_feat.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        output = self.decoder(feat_spatial)
        
        # 关键修复：应用可学习的logits缩放
        # 这将显著增加logits的数值范围，使模型能够做出更自信的预测
        output = output * self.logits_scale
        
        # 上采样到原始尺寸
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        # 返回结果
        if self.training and consistency_loss is not None:
            # 训练模式：返回分割结果和辅助损失
            aux_losses = {
                'consistency_loss': consistency_loss * self.consistency_loss_weight
            }
            return output, aux_losses
        else:
            # 推理模式：只返回分割结果
            return output


def create_acf_model(dataset: str, num_classes: int, **kwargs) -> ACFNetwork:
    """
    创建ACF模型的工厂函数
    
    Args:
        dataset: 'vaihingen', 'augsburg', 'muufl', 'trento'
        num_classes: 类别数
            - Vaihingen: 6 (Imp., Building, Low veg., Tree, Car, Clutter)
            - Augsburg: 7
            - MUUFL: 11
            - Trento: 6
    """
    return ACFNetwork(dataset=dataset, num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    # 测试各数据集
    print("Testing ACF Network on different datasets...")
    
    # 1. Vaihingen
    print("\n1. Vaihingen (RGB+DSM)")
    model_v = create_acf_model('vaihingen', num_classes=6)
    inputs_v = {
        'rgb': torch.randn(2, 3, 512, 512),
        'dsm': torch.randn(2, 1, 512, 512)
    }
    output_v = model_v(inputs_v)
    print(f"   Output shape: {output_v.shape}")
    
    # 2. Augsburg
    print("\n2. Augsburg (HSI+PolSAR+DSM)")
    model_a = create_acf_model('augsburg', num_classes=7)
    inputs_a = {
        'hsi': torch.randn(2, 180, 512, 512),
        'sar': torch.randn(2, 4, 512, 512),
        'dsm': torch.randn(2, 1, 512, 512)
    }
    output_a = model_a(inputs_a)
    print(f"   Output shape: {output_a.shape}")
    
    # 3. MUUFL
    print("\n3. MUUFL (HSI+LiDAR)")
    model_m = create_acf_model('muufl', num_classes=11)
    inputs_m = {
        'hsi': torch.randn(2, 64, 325, 220),
        'lidar': torch.randn(2, 2, 325, 220)
    }
    output_m = model_m(inputs_m)
    print(f"   Output shape: {output_m.shape}")
    
    # 4. Trento
    print("\n4. Trento (HSI+LiDAR)")
    model_t = create_acf_model('trento', num_classes=6)
    inputs_t = {
        'hsi': torch.randn(2, 63, 600, 166),
        'lidar': torch.randn(2, 1, 600, 166)
    }
    output_t = model_t(inputs_t)
    print(f"   Output shape: {output_t.shape}")
    
    print("\n✓ All tests passed!")
