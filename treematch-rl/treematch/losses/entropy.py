"""粒子熵正则化 (RBF 核斥力)

论文 §4.3:
    L_Entropy = (1/K(K-1)) Σ_{i≠j} exp(-||φ(x_i) - φ(x_j)||² / h)

在 latent 空间中, 当两条路径的最终生成图像语义过于接近时施加排斥力,
防止 27 个分支坍缩为同一解。
"""

import torch
import torch.nn as nn


class ParticleEntropyLoss(nn.Module):
    """基于 RBF 核函数的粒子熵正则化

    物理意义: 类似弹性碰撞, 当粒子间距离太近时产生斥力
    """

    def __init__(self, bandwidth: float = 1.0, feature_space: str = "latent"):
        """
        Args:
            bandwidth: RBF 核带宽 h (越大排斥力作用距离越远)
            feature_space: 计算距离的特征空间
                "latent" → 直接用 latent 特征
                "clip" → 用 CLIP 编码 (需外部提供)
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.feature_space = feature_space

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (K, D) 各路径在特征空间的表示
                K = 27 (路径数), D = 特征维度

        Returns:
            loss: scalar, 越大表示路径间越相似 (需要最小化)
        """
        features = features.float()
        K = features.shape[0]
        if K < 2:
            return torch.tensor(0.0, device=features.device)

        # 数值稳定: 批内标准化后再计算距离, 避免 RBF 指数项长期下溢到 0。
        feat_mean = features.mean(dim=0, keepdim=True)
        feat_std = features.std(dim=0, keepdim=True, unbiased=False)
        features = (features - feat_mean) / (feat_std + 1e-6)

        # 成对距离矩阵: (K, K)
        dists_sq = torch.cdist(features, features, p=2.0).pow(2)

        # RBF 核: exp(-d² / h)
        exponent = -dists_sq / (self.bandwidth + 1e-8)
        # float32 下 exp(x) 当 x << -80 时易直接下溢到 0。
        exponent = torch.clamp(exponent, min=-60.0, max=0.0)
        rbf = torch.exp(exponent)

        # 去掉对角线 (自身距离 = 0, exp(0) = 1)
        mask = ~torch.eye(K, dtype=torch.bool, device=features.device)
        loss = rbf[mask].mean()

        return loss

    @staticmethod
    def compute_latent_features(latents: torch.Tensor) -> torch.Tensor:
        """将 latent 转换为特征向量

        Args:
            latents: (K, C, H, W) 各路径的最终 latent

        Returns:
            features: (K, D) 扁平化后的特征
        """
        K = latents.shape[0]
        # 全局平均池化 → (K, C)
        features = latents.mean(dim=(-2, -1))
        return features
