from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.modules.encdec import *
from monai.utils import ensure_tuple_rep
from models.sam_encoder import ImageEncoder

class SynESAM(nn.Module):
    def __init__(
        self, input_size, in_ch, out_ch, base_filters=20, norm="instance", dropout=0.0, attention_dropout=0.0,
        stochastic_depth=0.0, apply_norm=True, dims=2, transformer_dim=1280, transformer_layers=32,
        transformer_heads=16, global_attention_layers=[7, 15, 23, 31], patch_dim=16,
        transformer_out_ch=256, freeze_transformer=False
    ):
        super().__init__()

        input_size = ensure_tuple_rep(input_size, dims)
        patch_dims = ensure_tuple_rep(patch_dim, dims)

        self.apply_norm = apply_norm
        self.project_shape = tuple(a // b for a, b in zip(input_size, patch_dims))
        self.reshape_dims = list(self.project_shape) + [transformer_dim]
        self.permute_order = (0, dims + 1) + tuple(i + 1 for i in range(dims))

        # === Vision Transformer ===
        self.transformer_encoder = ImageEncoder(
            depth=transformer_layers,
            embed_dim=transformer_dim,
            img_size=input_size[0],
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=transformer_heads,
            patch_size=patch_dim,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=global_attention_layers,
            window_size=14,
            out_chans=transformer_out_ch,
            in_chans=in_ch
        )

        # === Load and resize pretrained weights ===
        state_dict = torch.load(r"C:\Users\UAB\Box\Kim_Lab_CK\PhD\Prostate\Paper 2 SynSAM\models\checkpoints\sam_encoder.pth")
        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].mean(dim=1).unsqueeze(1)

        embed_size = input_size[0] // 16
        rel_shape = ((input_size[0] // 8) - 1, transformer_dim // 16)

        resize_map = {
            'pos_embed': (embed_size, embed_size),
            **{f'blocks.{i}.attn.rel_pos_h': rel_shape for i in global_attention_layers},
            **{f'blocks.{i}.attn.rel_pos_w': rel_shape for i in global_attention_layers},
        }

        for key, shape in resize_map.items():
            state_dict[key] = self._resize_tensor(state_dict[key], shape)

        self.transformer_encoder.load_state_dict(state_dict)

        if freeze_transformer:
            for p in self.transformer_encoder.parameters():
                p.requires_grad = False
            print("Transformer encoder frozen.")

        # === Convolution Blocks ===
        self.init_conv = ResInBlock(dims, in_ch, base_filters, 3, 1, norm)

        self.skip2 = ResUNetSkipEncoder(dims, transformer_dim, base_filters * 2, 2, 3, 1, 2, norm)
        self.skip3 = ResUNetSkipEncoder(dims, transformer_dim, base_filters * 4, 1, 3, 1, 2, norm)
        self.skip4 = ResUNetSkipEncoder(dims, transformer_dim, base_filters * 8, 0, 3, 1, 2, norm)

        self.decode4 = ResUNetSkipDecoder(dims, transformer_dim, base_filters * 8, 3, 2, norm)
        self.decode3 = ResUNetSkipDecoder(dims, base_filters * 8, base_filters * 4, 3, 2, norm)
        self.decode2 = ResUNetSkipDecoder(dims, base_filters * 4, base_filters * 2, 3, 2, norm)
        self.decode1 = ResUNetSkipDecoder(dims, base_filters * 2, base_filters, 3, 2, norm)

        self.segmentation_head = ConvOut(dims, base_filters, out_ch)

    def _resize_tensor(self, tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        reshaped = tensor.unsqueeze(0).unsqueeze(-1) if tensor.dim() == 2 else tensor
        reshaped = reshaped.permute(0, -1, *range(1, reshaped.dim() - 1))
        resized = F.interpolate(reshaped, size=target_size, mode='bilinear')
        resized = resized.permute(0, 2, 3, 1)
        return resized.squeeze() if tensor.dim() == 2 else resized

    def _reshape_and_permute(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size = tensor.size(0)
        reshaped = tensor.view(batch_size, *self.reshape_dims)
        return reshaped.permute(self.permute_order).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_out, feat_list = self.transformer_encoder(x)

        stage0 = self.init_conv(x)
        emb7 = self._reshape_and_permute(feat_list[7])
        emb15 = self._reshape_and_permute(feat_list[15])
        emb23 = self._reshape_and_permute(feat_list[23])
        emb31 = self._reshape_and_permute(feat_list[31])

        skip2_out = self.skip2(emb7)
        skip3_out = self.skip3(emb15)
        skip4_out = self.skip4(emb23)

        bottleneck = emb31

        up4 = self.decode4(bottleneck, skip4_out)
        up3 = self.decode3(up4, skip3_out)
        up2 = self.decode2(up3, skip2_out)
        up1 = self.decode1(up2, stage0)

        output = self.segmentation_head(up1)
        return output
