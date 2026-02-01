import torch
import torch.nn as nn

from models.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from models.common.utils import SinusoidalPosEmb, init_weights
from .base_policy import NoisePredictionNet
import ipdb


class MotionProjector(nn.Module):
    def __init__(self, embed_dim, hidden_dim=512):
        super().__init__()
        # 第一步：先映射到一个足以支撑 14x14 分辨率的特征空间
        # 14 * 14 = 196
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 14 * 14 * 16), # 16是中间通道数
        )
        
        # 第二步：空间维度细化
        self.conv_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv2d(16, 2, kernel_size=1) # 最终输出 2 通道 (dx, dy)
        )

    def forward(self, motion_feats):
        # motion_feats shape: (B, motion_len, embed_dim)
        b, l, d = motion_feats.shape
        
        # 映射并展开空间维度
        x = self.fc(motion_feats) # (B, L, 14*14*16)
        x = x.view(b * l, 16, 14, 14) # 准备进入 CNN 处理空间相关性
        
        # 局部特征优化
        out = self.conv_head(x) # (B*L, 2, 14, 14)
        
        # 还原回你的 GT 形状: (B, L, 14, 14, 2)
        out = out.permute(0, 2, 3, 1).view(b, l, 14, 14, 2)
        return out
    

class TransformerNoisePredictionNet(NoisePredictionNet):
    def __init__(
        self,
        input_len: int,
        input_dim: int,
        global_cond_dim: int,
        timestep_embed_dim: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_motion_token: bool = False,
        motion_mask: bool = False,
    ):
        super().__init__()
        print("*"*30)
        print("use_motion_token:", use_motion_token)
        print("*"*30)
        self.use_motion_token = use_motion_token
        self.input_len = input_len
        self.motion_mask = motion_mask
        # Input encoder and decoder
        hidden_dim = int(max(input_dim, embed_dim) * mlp_ratio)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(), #Mish激活
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Timestep encoder
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        # Model components
        self.pos_embed = nn.Parameter(
            torch.empty(1, input_len, embed_dim).normal_(std=0.02)
        )

        if self.use_motion_token:
            #NOTE:加入motion-token
            self.motion_tokens = nn.Parameter(torch.zeros(1, input_len, embed_dim))
            nn.init.normal_(self.motion_tokens, std=0.02)
            #NOTE：加入motion-token的映射层
            #FIXME:这里的hidden_dim和上面input_encoder的hidden_dim是一样的
            self.motion_projector = MotionProjector(embed_dim=embed_dim, hidden_dim=hidden_dim)
        
        cond_dim = global_cond_dim + timestep_embed_dim #vision token和时间步
        
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,#用时间和vision作为condition
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)

        # AdaLN-specific weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Base initialization
        self.apply(init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(self, sample, timestep, global_cond):
        #sample是noisy action，global_cond是obs
        #后续拼接上motion token
        #sample.shape torch.Size([1, 16, 7])
        # Encode input
        b = sample.shape[0]
        l_a = sample.shape[1]

        embed = self.input_encoder(sample)

        # Encode timestep
        if len(timestep.shape) == 0:
            timestep = timestep.expand(sample.shape[0]).to(
                dtype=torch.long, device=sample.device
            )
        temb = self.timestep_encoder(timestep)

        # Concatenate timestep and condition along the sequence dimension
        x = embed + self.pos_embed

        attn_mask=None
        if self.use_motion_token:
            #NOTE：加入motiontoken的传播
            m_tokens = self.motion_tokens.expand(b, -1, -1)
            l_m = m_tokens.shape[1]
            x = torch.cat([x, m_tokens], dim=1)
            l_total = l_a + l_m
            mask = torch.ones((l_total,l_total),device=sample.device,dtype=torch.bool)
            if self.motion_mask:
                mask[l_a:, :l_a] = False
                attn_mask = mask.unsqueeze(0).repeat(b, 1, 1)
        
        cond = torch.cat([global_cond, temb], dim=-1)

        #NOTE:取倒数第二层的mv出来做对齐
        intermediate_output = None
        for i, block in enumerate(self.blocks):
            x = block(x, cond, attn_mask=attn_mask)
            if self.use_motion_token and i == len(self.blocks) - 2:  # 倒数第二个 Block
                intermediate_output = x
        
        pred_mvs = None
        if self.use_motion_token:
            motion_feats = intermediate_output[:, -self.input_len :, :]
            pred_mvs = self.motion_projector(motion_feats)
            # 这里的 x 需要截断，只保留 action 部分进入 head
            action_out = x[:, : self.input_len, :]
        else:
            action_out = x

        x = self.head(action_out, cond)
        out = self.output_decoder(x)
        return out,pred_mvs
