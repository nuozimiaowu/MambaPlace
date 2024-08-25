from mamba_model import Mamba2
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.language_encoder import LanguageEncoder
from models.object_encoder import ObjectEncoder
from models.new import create_block
import math

class RowColRPA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(RowColRPA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.r_linear = nn.Linear(embed_dim, embed_dim)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, instance_features):
        batch_size, num_instances, embed_dim = instance_features.shape

        # 计算 Q, K, V
        Q = self.q_linear(instance_features)
        K = self.k_linear(instance_features)
        V = self.v_linear(instance_features)

        # 计算相对位置
        R = instance_features[:, :, None] - instance_features[:, None, :]
        R = self.r_linear(R)

        max_R1 = torch.max(R, dim=1, keepdim=True).values
        max_R1_squeezed = torch.squeeze(max_R1, dim=1)
        Q_r = Q + max_R1_squeezed

        max_R2 = torch.max(R, dim=2, keepdim=True).values
        max_R2_squeezed = torch.squeeze(max_R2, dim=2)
        K_r = K + max_R2_squeezed

        # 计算多头注意力
        Q_r = Q_r.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(1, 2)
        K_r = K_r.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q_r, K_r.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, num_instances, embed_dim)

        # 残差连接和层归一化
        attention_output = self.dropout(self.out(attention_output))
        instance_features = self.layer_norm1(instance_features + attention_output)

        # 第二层残差连接和层归一化
        output = self.layer_norm2(instance_features)
        return output



class CellRetrievalNetwork(torch.nn.Module):
    def __init__(
            self, known_classes: List[str], known_colors: List[str], args
    ):
        """Module for global place recognition.
        Implemented as a text branch (language encoder) and a 3D submap branch (object encoder).
        The 3D submap branch aggregates information about a varying count of multiple objects through Attention.
        """
        super(CellRetrievalNetwork, self).__init__()
        self.embed_dim = args.coarse_embed_dim
        self.mamba_layer = Mamba2(args.coarse_embed_dim)
        """
        3D submap branch
        """

        # CARE: possibly handle variation in forward()!
        self.object_encoder = ObjectEncoder(args.coarse_embed_dim, known_classes, known_colors, args)
        self.object_size = args.object_size
        num_heads = args.object_inter_module_num_heads
        num_layers = args.object_inter_module_num_layers

        # 使用RowColRPA模块替换原有的TransformerEncoderLayer模块
        self.row_col_rpa = RowColRPA(args.coarse_embed_dim, num_heads)

        self.layers = create_block(
            d_model=256,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            rms_norm=False,
            residual_in_fp32=False,
            fused_add_norm=False,
            layer_idx=None,
            drop_path=0.1,
            device='cuda',
            dtype=torch.float32,
        )
        self.obj_inter_module = nn.ModuleList([nn.TransformerEncoderLayer(args.coarse_embed_dim, num_heads,  dim_feedforward = 2 * args.coarse_embed_dim) for _ in range(num_layers)])
        """
        Textual branch
        """
        self.language_encoder = LanguageEncoder(args.coarse_embed_dim,
                                                hungging_model=args.hungging_model,
                                                fixed_embedding=args.fixed_embedding,
                                                intra_module_num_layers=args.intra_module_num_layers,
                                                intra_module_num_heads=args.intra_module_num_heads,
                                                is_fine=False,
                                                inter_module_num_layers=args.inter_module_num_layers,
                                                inter_module_num_heads=args.inter_module_num_heads,
                                                )

        print(
            f"CellRetrievalNetwork, class embed {args.class_embed}, color embed {args.color_embed}, dim: {args.coarse_embed_dim}, features: {args.use_features}"
        )


    def encode_text(self, descriptions):

        description_encodings = self.language_encoder(descriptions)  # [B, DIM]

        description_encodings = F.normalize(description_encodings)

        return description_encodings

    def encode_objects(self, objects, object_points):
        """
        Process the objects in a flattened way to allow for the processing of batches with uneven sample counts
        """

        batch = []  # Batch tensor to send into PyG
        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:
                # class_idx = self.known_classes.get(obj.label, 0)
                # class_indices.append(class_idx)
                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        embeddings, pos_postions = self.object_encoder(objects, object_points)

        object_size = self.object_size

        index_list = [0]
        last = 0

        x = torch.zeros(len(objects), object_size, self.embed_dim).to(self.device)

        for obj in objects:
            index_list.append(last + len(obj))
            last += len(obj)

        embeddings = F.normalize(embeddings, dim=-1)

        for idx in range(len(index_list) - 1):
            num_object_raw = index_list[idx + 1] - index_list[idx]
            start = index_list[idx]
            num_object = num_object_raw if num_object_raw <= object_size else object_size
            x[idx, : num_object] = embeddings[start: (start + num_object)]

        x,x_residual = self.layers(x)

        x = x.permute(1, 0, 2).contiguous()

        del embeddings, pos_postions

        x = x.max(dim=0)[0]
        x = F.normalize(x)

        return x

    def forward(self):
        raise Exception("Not implemented.")

    @property
    def device(self):
        return self.language_encoder.device

    def get_device(self):
        return self.language_encoder.device
