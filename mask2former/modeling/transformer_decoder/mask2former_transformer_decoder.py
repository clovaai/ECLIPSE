"""
ECLIPSE
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
The implementation is based on facebookresearch/Mask2Former.
"""

import logging
from copy import deepcopy
import fvcore.nn.weight_init as weight_init
from typing import Optional, List
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from continual.modeling import IncrementalClassifier, CosineClassifier


def sigmoid_to_logit(x):
    x = x.clamp(0.001, 0.999)
    return torch.log(x / (1-x))

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        num_prompts: int,
        prompt_deep: bool = False,
        softmask: bool = False,
        inc_query: Optional[bool] = None,
        cosine: Optional[bool] = False,
        bias: Optional[bool] = False,
        classes: Optional[List[int]] = None,
        prompt_mask_mlp: Optional[bool] = False,
        prompt_no_obj_mlp: Optional[bool] = False,
        deep_cls: Optional[bool] = False,
        deltas: Optional[List[float]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.softmask = softmask

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_classes = num_classes
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Parameters for ECLIPSE
        self.num_prompts = num_prompts
        self.prompt_deep = prompt_deep and self.num_prompts > 0
        self.prompt_mask_mlp = prompt_mask_mlp and self.num_prompts > 0
        self.prompt_no_obj_mlp = prompt_no_obj_mlp and self.num_prompts > 0
        
        self.deltas = deltas
        if self.deltas is None:
            self.deltas = [0.0 for _ in classes]
        elif type(self.deltas) == float:
            self.deltas = [self.deltas for _ in classes]
        elif len(self.deltas) > len(classes):
            self.deltas = self.deltas[:len(classes)]
        elif len(self.deltas) < len(classes):
            self.deltas = self.deltas + [self.deltas[-1] for _ in range(len(classes)-len(self.deltas))]
            
        assert len(self.deltas) == len(classes), "CONT."
        
        self.old_model = False
        
        # prompt embeddings
        if self.num_prompts > 0:
            self.prompt_feat = nn.ModuleList(
                [nn.Embedding(num_prompts, hidden_dim) for _ in classes[1:]]
            )
            
            if self.prompt_deep:
                self.prompt_embed = nn.ModuleList(
                    [
                        nn.ModuleList(
                            [nn.Embedding(num_prompts, hidden_dim) for _ in range(self.num_layers)]
                        ) for _ in classes[1:]
                    ]
                )
                
            else:
                self.prompt_embed = nn.ModuleList(
                    [nn.Embedding(num_prompts, hidden_dim) for _ in classes[1:]]
                )

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.num_classes = num_classes
        self.classes = classes # [C0, C1, ..., Cn]
        
        if self.mask_classification:
            if classes is not None:
                if cosine:
                    self.class_embed = CosineClassifier([1] + classes, channels=hidden_dim)
                else:
                    # [1] : no_obj, (we don't have bkg class)
                    self.class_embed = IncrementalClassifier(
                        [1] + classes, 
                        channels=hidden_dim, 
                        bias=bias, 
                        deep_cls=deep_cls,
                    )
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
                
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        
        if self.prompt_mask_mlp:
            self.prompt_mask_embed = nn.ModuleList(
                [deepcopy(self.mask_embed) for _ in classes[1:]]
            )
            
        if self.prompt_no_obj_mlp:
            self.prompt_no_obj_embed = nn.ModuleList(
                [MLP(hidden_dim, hidden_dim, 1, 3) for _ in classes[1:]]
            )
            
    def set_as_old_model(self, ):
        self.old_model = True
        self.prompt_feat = None
        self.prompt_embed = None
        self.prompt_mask_mlp = False
        self.prompt_no_obj_mlp = False

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["softmask"] = cfg.MODEL.MASK_FORMER.SOFTMASK

        if hasattr(cfg, "CONT"):
            ret['inc_query'] = cfg.CONT.INC_QUERY
            ret["classes"] = [cfg.CONT.BASE_CLS] + cfg.CONT.TASK*[cfg.CONT.INC_CLS]
            ret["num_classes"] = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
            ret["cosine"] = cfg.CONT.COSINE
            ret["bias"] = cfg.CONT.USE_BIAS
            if cfg.MODEL.MASK_FORMER.TEST.MASK_BG and (cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON or
                                                       cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON):
                ret["num_classes"] += 1
                ret["classes"][0] += 1
            
            # Parameters for ECLIPSE
            ret['num_prompts'] = cfg.CONT.NUM_PROMPTS
            ret['prompt_deep'] = cfg.CONT.PROMPT_DEEP
            ret['prompt_mask_mlp'] = cfg.CONT.PROMPT_MASK_MLP
            ret['prompt_no_obj_mlp'] = cfg.CONT.PROMPT_NO_OBJ_MLP
            ret['deltas'] = cfg.CONT.LOGIT_MANI_DELTAS
            ret['deep_cls'] = cfg.CONT.DEEP_CLS
            
        else:
            ret['inc_query'] = None
            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            if not cfg.MODEL.MASK_FORMER.TEST.MASK_BG and (cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON or
                                                       cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON):
                ret["num_classes"] -= 1
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret
    
    
    def forward_base_train(self, x, src, pos, size_list, mask_features):
        """
        Training the model for base classes (before applying visual prompt tuning, step 0)
        """
        
        _, bs, _ = src[0].shape
        predictions_class = []
        predictions_mask = []
        
        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, self.class_embed, self.mask_embed, mask_features, 
            attn_mask_target_size=size_list[0], 
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, 
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, self.class_embed, self.mask_embed, mask_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
            )

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            # 'query': query_feat,
            'features': mask_features
        }
        return out
    
            
    def forward_new_train(self, x, src, pos, size_list, mask_features):
        """
        Training the model for novel classes (applying visual prompt tuning, step > 1)
        """
            
        _, bs, _ = src[0].shape
        predictions_class = []
        predictions_mask = []

        if self.num_prompts > 0 and self.prompt_no_obj_mlp:
            query_dims = np.cumsum([0, self.num_queries] + [self.num_prompts] * len(self.prompt_embed))
        else:
            query_dims = None
            
        output_p = self.prompt_feat[-1].weight.unsqueeze(1).repeat(1, bs, 1)

        # prediction heads on learnable query features
        outputs_class_p, outputs_mask_p, attn_mask_p = self.forward_prediction_heads(
            output_p, 
            self.class_embed, 
            self.mask_embed if not self.prompt_mask_mlp else self.prompt_mask_embed[-1],
            mask_features, 
            attn_mask_target_size=size_list[0],
            qdims=query_dims,
        )

        predictions_class.append(outputs_class_p)
        predictions_mask.append(outputs_mask_p)

        for i in range(self.num_layers):
            if self.prompt_deep:
                prompt_embed = self.prompt_embed[-1][i].weight.unsqueeze(1).repeat(1, bs, 1)
            else:
                prompt_embed = self.prompt_embed[-1].weight.unsqueeze(1).repeat(1, bs, 1)

            level_index = i % self.num_feature_levels
            attn_mask_p[torch.where(attn_mask_p.sum(-1) == attn_mask_p.shape[-1])] = False

            # attention: cross-attention first
            output_p = self.transformer_cross_attention_layers[i](
                output_p, 
                src[level_index],
                memory_mask=attn_mask_p,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=prompt_embed
            )

            output_p = self.transformer_self_attention_layers[i](
                output_p, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=prompt_embed
            )

            # FFN
            output_p = self.transformer_ffn_layers[i](
                output_p
            )

            outputs_class_p, outputs_mask_p, attn_mask_p = self.forward_prediction_heads(
                output_p, 
                self.class_embed, 
                self.mask_embed if not self.prompt_mask_mlp else self.prompt_mask_embed[-1],
                mask_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                qdims=query_dims,
            )

        predictions_class.append(outputs_class_p)
        predictions_mask.append(outputs_mask_p)
        
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            # 'query': query_feat,
            'features': mask_features
        }
        return out

    
    def forward_infer(self, x, src, pos, size_list, mask_features):
        """
        Inference for ECLIPSE
        """
        _, bs, _ = src[0].shape
        stacked_dec = range(len(self.classes)) if self.num_prompts > 0 else range(1)
        
        predictions_class = []
        predictions_mask = []
        
        if self.num_prompts > 0:
            mask_embeds = nn.ModuleList([self.mask_embed])
            mask_embeds = mask_embeds.extend(self.prompt_mask_embed)
            query_dims = np.cumsum([0, self.num_queries] + [self.num_prompts] * len(self.prompt_embed))
        else:
            mask_embeds = self.mask_embed
            query_dims = None
        
        # QxNxC
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        
        if self.num_prompts > 0:
            output = torch.cat(
                [
                    output, 
                    torch.cat([p.weight.unsqueeze(1).repeat(1, bs, 1) for p in self.prompt_feat], dim=0)
                ], dim=0
            )
        
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, 
            self.class_embed, 
            mask_embeds,
            mask_features, 
            attn_mask_target_size=size_list[0], 
            qdims=query_dims,
        )
        
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.num_prompts > 0:
                if self.prompt_deep:
                    query_embed = torch.cat(
                        [
                            query_embed, 
                            torch.cat([p[i].weight.unsqueeze(1).repeat(1, bs, 1) for p in self.prompt_embed], dim=0)
                        ], dim=0
                    )
                else:
                    query_embed = torch.cat(
                        [
                            query_embed, 
                            torch.cat([p.weight.unsqueeze(1).repeat(1, bs, 1) for p in self.prompt_embed], dim=0)
                        ], dim=0
                    )
            
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, 
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            
            if self.num_prompts > 0:
                self_attn_outputs = torch.zeros_like(output)
                for qn, qdim in enumerate(query_dims[:-1]):
                    self_attn_outputs[query_dims[qn]:query_dims[qn+1]] = self.transformer_self_attention_layers[i](
                        output[query_dims[qn]:query_dims[qn+1]], tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed[query_dims[qn]:query_dims[qn+1]]
                    )
                output = self_attn_outputs
                
            else:
                output = self.transformer_self_attention_layers[i](
                        output, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed,
                    )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, self.class_embed, mask_embeds, mask_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                qdims=query_dims,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            #'aux_outputs': self._set_aux_loss(
            #    predictions_class if self.mask_classification else None, predictions_mask
            #),
            # 'query': query_feat,
            #'features': mask_features
        }
        return out
    
    
    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        if not self.training:
            return self.forward_infer(x, src, pos, size_list, mask_features)
        else:
            if self.num_prompts > 0 and not self.old_model:
                return self.forward_new_train(x, src, pos, size_list, mask_features)
                
            else:
                return self.forward_base_train(x, src, pos, size_list, mask_features)
            
            
    def forward_prediction_heads(self, output, class_embed, mask_embed, mask_features, 
                                 attn_mask_target_size, qdims=None):
        
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        
        outputs_class = class_embed(decoder_output) # outputs_class : [B, 100, 100+10+1]
        
        # logit manipulation implementation
        if not self.training and self.num_prompts > 0 and qdims is not None:
            m_embed = []
            for n in range(len(qdims)-1):
                m_embed.append(mask_embed[n](decoder_output[:, qdims[n]:qdims[n+1]]))
                
                if self.prompt_no_obj_mlp and n > 0:
                    no_obj_logit = self.prompt_no_obj_embed[n-1](decoder_output)
                    outputs_class[:, qdims[n]:qdims[n+1], -1] = no_obj_logit[:, qdims[n]:qdims[n+1], 0]
                    
                if self.deltas[n] > 0:
                    # logit manipulation with delta: aggregation of other class knowledge
                    noobj_score = outputs_class[:, qdims[n]:qdims[n+1], 
                                                list(range(0, sum(self.classes[:n]))) + \
                                                list(range(sum(self.classes[:n+1]), sum(self.classes)))
                                               ].sigmoid().sum(2).clamp(0., 1.)

                    outputs_class[:, qdims[n]:qdims[n+1], -1] = sigmoid_to_logit(
                        noobj_score * self.deltas[n]
                    )
                    
                elif self.deltas[n] < 0:
                    # negative delta means calibration the class logits without aggregation of other class knowledge
                    # we empirically found that this strategy is effective when the number of incremental steps is small (e.g., 100-50).
                    outputs_class[:, qdims[n]:qdims[n+1], -1] = sigmoid_to_logit(
                        outputs_class[:, qdims[n]:qdims[n+1], -1].sigmoid() * -self.deltas[n]
                    )
                        
                # deactivate other class logits: regarding sigmoid(-10) => 0.0
                outputs_class[:, qdims[n]:qdims[n+1], 
                              list(range(0, sum(self.classes[:n]))) + \
                              list(range(sum(self.classes[:n+1]), sum(self.classes)))
                             ] = -10
                    
            m_embed = torch.cat(m_embed, dim=1)
            
        else:
            m_embed = mask_embed(decoder_output)
            if self.prompt_no_obj_mlp and qdims is not None:
                for n in range(1, len(qdims)-1):
                    no_obj_logit = self.prompt_no_obj_embed[n-1](decoder_output)
                    outputs_class[:, qdims[n]:qdims[n+1], -1] = no_obj_logit[:, qdims[n]:qdims[n+1], 0]
            
        outputs_masks = torch.einsum("bqc,bchw->bqhw", m_embed, mask_features)
            
        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_masks, size=attn_mask_target_size, mode="bilinear", align_corners=False)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        if self.softmask:
            attn_mask = (attn_mask.softmax(dim=1).flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        else:
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_masks, attn_mask
    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
