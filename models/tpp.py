"""
model class.
"""
import torch
import torch.nn.functional as F
from torch import nn

import os
import math
from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer_plus import build_deforamble_transformer
from .segmentation import CrossModalFPNDecoder, VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast, AutoTokenizer, AutoModel

import copy
from einops import rearrange, repeat
from numpy import random


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

class TPP(nn.Module):

    def __init__(self, args, backbone, transformer, criterion, num_classes, num_queries, num_feature_levels,
                 num_frames, mask_dim, dim_feedforward,
                 controller_layers, dynamic_mask_channels,
                 aux_loss=False, with_box_refine=False, two_stage=False,
                 freeze_text_encoder=False, rel_coord=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot.
            num_frames:  number of clip frames
            mask_dim: dynamic conv inter layer channel number.
            dim_feedforward: vision-language fusion module ffn channel number.
            dynamic_mask_channels: the mask feature output channel number.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.criterion = criterion
        self.update_pos = args.update_pos

        # Build Transformer
        # NOTE: different deformable detr, the query_embed out channels is
        # hidden_dim instead of hidden_dim * 2
        # This is because, the input to the decoder is text embedding feature
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # follow deformable-detr, we use the last three stages of backbone
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):  # downsample 2x
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.mask_dim = mask_dim
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert two_stage == False, "args.two_stage must be false!"

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # Build Text Encoder

        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # resize the bert output channel to transformer d_model
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)

        # Build FPN Decoder
        self.rel_coord = rel_coord
        feature_channels = [self.backbone.num_channels[0]] + 3 * [hidden_dim]
        self.pixel_decoder = CrossModalFPNDecoder(feature_channels=feature_channels, conv_dim=hidden_dim,
                                                  mask_dim=mask_dim, dim_feedforward=dim_feedforward, norm="GN")

        # Build Dynamic Conv
        self.controller_layers = controller_layers
        self.in_channels = mask_dim
        self.dynamic_mask_channels = dynamic_mask_channels
        self.mask_out_stride = 4
        self.mask_feat_stride = 4

        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)  # output layer c -> 1
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)

        # Here TrackEmbedding is MLPs
        self.track_embed = TrackEmbedding(hidden_dim, hidden_dim, hidden_dim)
        self.use_checkpoint_for_more_frames = args.use_checkpoint_for_more_frames
        if self.update_pos:
            self.track_pos_embed = TrackEmbedding(hidden_dim, hidden_dim, hidden_dim)

    def generate_empty_tracks(self,):

        num_queries, dim = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device

        track_res = {
            'ref_pts': self.transformer.reference_points(self.query_embed.weight),
            'track_embedding': torch.zeros((num_queries, dim), device=device),
            'pos_embedding': self.query_embed.weight,
            'track_mask': torch.zeros((num_queries, dim)),
            'track_memory': torch.zeros((num_queries, dim))
        }

        return track_res

    def forward_single_frame(self, samples: NestedTensor, track_res, captions, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
               - captions: list[str]
               - targets:  list[dict]

            It returns a dict with the following elements:
               - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]

               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        # features (list[NestedTensor]): res2 -> res5, shape of tensors is [B*T, Ci, Hi, Wi]
        # pos (list[Tensor]): shape of [B*T, C, Hi, Wi]
        features, pos = self.backbone(samples)  # 1/4,1/8,1/16,1/32

        b = len(captions)
        t = pos[0].shape[0] // b

        if 'valid_indices' in targets[0]:
            valid_indices = torch.tensor([i * t + target['valid_indices'] for i, target in enumerate(targets)]).to(
                pos[0].device)
            for feature in features:
                feature.tensors = feature.tensors.index_select(0, valid_indices)
                feature.mask = feature.mask.index_select(0, valid_indices)
            for i, p in enumerate(pos):
                pos[i] = p.index_select(0, valid_indices)
            samples.mask = samples.mask.index_select(0, valid_indices)
            # t: num_frames -> 1
            t = 1

        text_features, text_sentence_features = self.forward_text(captions, device=pos[0].device)

        # prepare vision and text features for transformer
        srcs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size->3, c]
        text_word_features, text_word_masks = text_features.decompose()

        text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]

        # Follow Deformable-DETR, we use the last three stages outputs from backbone
        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape

            # vision language early-fusion
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            src_proj_l, _ = self.fusion_module(tgt=src_proj_l,
                                            memory=text_word_features,
                                            memory_key_padding_mask=text_word_masks,
                                            pos=text_pos,
                                            query_pos=None,
                                            h=h, w=w
                                            )
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1  # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                src, weight_index = self.fusion_module(tgt=src,
                                         memory=text_word_features,
                                         memory_key_padding_mask=text_word_masks,
                                         pos=text_pos,
                                         query_pos=None,
                                         h=h, w=w
                                         )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        # Transformer
        query_embeds = track_res['pos_embedding']  # torch.Size([5, 256])

        # query propagation
        if track_res['track_embedding'].sum() != 0:
            text_embed = track_res['track_embedding'].unsqueeze(0).unsqueeze(0)  # torch.Size([1, 1, 5, 256])
        else:
            # text_sentence_features 3,256 -> 1,1,5,256
            text_embed = text_sentence_features[weight_index, :].unsqueeze(0).unsqueeze(0)
            text_embed = repeat(text_embed, 'b t c -> b t q c', q=track_res['pos_embedding'].shape[0])

        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples, track_memory_current = \
            self.transformer(srcs, text_embed, masks, poses, query_embeds, track_res['ref_pts'], track_res['track_mask'], track_res['track_memory'])

        out = {}
        # prediction
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1]  # [batch_size, time, num_queries_per_frame, num_classes]
        out['pred_boxes'] = outputs_coord[-1]  # [batch_size, time, num_queries_per_frame, 4]

        # Segmentation
        mask_features = self.pixel_decoder(features, text_features, pos, memory, weight_index, nf=t)  # [batch_size*time, c, out_h, out_w]
        mask_features = rearrange(mask_features, '(b t) c h w -> b t c h w', b=b, t=t)

        # dynamic conv
        # mask_features: 1x1x256x96x160 -> repeat for 1x1x5x256x96x160 -> using dynamic_mask_head_params
        # -> outputs_seg_mask: 1x1x5x96x160
        # dynamic_mask_head_params: 1x5x2153 (2153 represents three convs)
        outputs_seg_masks = []
        for lvl in range(hs.shape[0]):
            dynamic_mask_head_params = self.controller(hs[lvl])  # [batch_size*time, num_queries_per_frame, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, '(b t) q n -> b (t q) n', b=b, t=t)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=b, t=t)
            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references,
                                                             targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> b t q h w', t=t)
            outputs_seg_masks.append(outputs_seg_mask)
        out['pred_masks'] = outputs_seg_masks[-1]  # [batch_size, time, num_queries_per_frame, out_h, out_w]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_masks)

        if not self.training:
            # for visualization
            init_references = init_reference[..., :4]
            init_references = rearrange(init_references, '(b t) q n -> b t q n', b=b, t=t)
            out['reference_inits'] = init_references  # the reference points of last layer input
            inter_references = inter_references[-2, :, :, :2]  # [batch_size*time, num_queries_per_frame, 2]
            inter_references = rearrange(inter_references, '(b t) q n -> b t q n', b=b, t=t)
            out['reference_points'] = inter_references  # the reference points of last layer input
        out['hs'] = hs[-1]
        out['pos_embedding'] = query_embeds
        
        return out, track_memory_current

    def post_process_single_image(self, frame_res, track_res, is_last, track_memory=None):

        pred_logits = frame_res['pred_logits'][0] #.detach().clone()
        video_len, query_len, _ = pred_logits.size()
        pred_scores = pred_logits.sigmoid()  # [t, q, k]
        pred_scores = pred_scores.mean(0)  # [q, k]
        max_scores, _ = pred_scores.max(-1)  # [q,]
        _, max_ind = max_scores.max(-1)  # [1,]
        max_inds = max_ind.repeat(video_len)

        # update reference boxes in track
        pred_boxes = frame_res['pred_boxes'][0][range(video_len), max_inds, ...]
        pred_boxes = pred_boxes.view([video_len, -1, 4]).detach().clone()
        track_res['ref_pts'] = inverse_sigmoid(pred_boxes[0])
        # no bbox propagation

        track_res['track_memory'] = track_memory  # 1, h/8, w/8, c

        # update mask
        pred_masks = frame_res['pred_masks'][0].detach().clone()
        pred_masks = pred_masks.view([video_len, -1, pred_masks.size()[-2], pred_masks.size()[-1]])
        track_res['track_mask'] = pred_masks[0]
        track_res['track_mask'] = track_res['track_mask'][max_inds, ...].unsqueeze(0)  # bs, 1, h/4, w/4
        # downsample to 1/2
        track_res['track_mask'] = F.interpolate(track_res['track_mask'], (track_memory.shape[1], track_memory.shape[2]), mode='bilinear', align_corners=False)

        # update query and position embedding in track
        output_embedding = frame_res['hs'][range(video_len), max_inds, ...]  # .detach().clone()
        output_embedding = output_embedding.view([video_len, -1, output_embedding.size()[-1]])
        if not is_last:
            output_embedding = self.track_embed(output_embedding)
            track_res['track_embedding'] = output_embedding[0]
            if self.update_pos:
                track_res['pos_embedding'] = self.track_pos_embed(frame_res['pos_embedding'][max_inds, ...])
            else:
                track_res['pos_embedding'] = frame_res['pos_embedding'][max_inds, ...]
        else:
            track_res['track_embedding'] = None
            track_res['pos_embedding'] = None

        return track_res

    def inference(self, samples:NestedTensor, track_res, captions, targets):
        
        if samples is not NestedTensor:
            frame = nested_tensor_from_videos_list(samples)
        
        frame_res, track_memory = self.forward_single_frame(frame, track_res, captions, targets)

        return frame_res, track_memory

    def forward(self, samples:NestedTensor, captions, targets):

        num_frames = samples.tensors.size()[1]
        track_res = self.generate_empty_tracks()

        loss_dicts = []
        keys = track_res.keys()
        for frame_index in range(num_frames):
            frame = samples.tensors[:, frame_index].unsqueeze(1)
            frame.requires_grad = False

            is_last = frame_index == num_frames - 1
            target_i = {k: targets[0][k][frame_index].unsqueeze(0) for k in targets[0].keys() if (k != 'orig_size' and k !='size' and k != 'yolov7_det')}
            target_i['orig_size'] = targets[0]['orig_size']
            target_i['size'] = targets[0]['size']
            target_i = [target_i]

            if self.use_checkpoint_for_more_frames and frame_index < num_frames - 1:
                def fn(frame, *args):
                    frame = nested_tensor_from_videos_list(frame)
                    tmp = dict(zip(keys, args))
                    frame_res, track_memory = self.forward_single_frame(frame, tmp, captions, target_i)
                    return (
                        track_memory,
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['pred_masks'],
                        frame_res['hs'],
                        frame_res['pos_embedding'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_masks'] for aux in frame_res['aux_outputs']]
                    )
                args = [frame] + [track_res.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                track_memory = tmp[0]
                frame_res = {
                    'pred_logits': tmp[1],
                    'pred_boxes': tmp[2],
                    'pred_masks': tmp[3],
                    'hs': tmp[4],
                    'pos_embedding': tmp[5],
                    'aux_outputs': [{
                        'pred_logits': tmp[6 + i],
                        'pred_boxes': tmp[6 + 3 + i],
                        'pred_masks': tmp[6 + 3 + 3 + i]
                    } for i in range(3)],
                }
            else:
                frame = nested_tensor_from_videos_list(frame)
                frame_res, track_memory = self.forward_single_frame(frame, track_res, captions, target_i)
            # prop
            track_res = self.post_process_single_image(frame_res, track_res, is_last, track_memory)

            loss_dict, _ = self.criterion(frame_res, target_i)
            loss_dicts.append(loss_dict)

        loss_dict_new = {k:0.0 for k in loss_dicts[0].keys()}
        for loss_dict in loss_dicts:
            for k in loss_dict.keys():
                loss_dict_new[k] += loss_dict[k]/num_frames
        return loss_dict_new

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_masks[:-1])]

    def forward_text(self, captions, device):
        # captions: list[list[str]]
        captions = captions[0]

        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length]

            text_features = encoded_text.last_hidden_state
            text_features = self.resizer(text_features)
            text_masks = text_attention_mask

            text_features = NestedTensor(text_features, text_masks)  # NestedTensor

            text_sentence_features = encoded_text.pooler_output
            text_sentence_features = self.resizer(text_sentence_features)
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features, text_sentence_features

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        # this is the total query number in all frames
        _, num_queries = reference_points.shape[:2]
        q = num_queries // t  # num_queries_per_frame

        # prepare reference points in image size (the size is input size to the model)
        new_reference_points = []
        for i in range(b):
            img_h, img_w = targets[i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0)
            tmp_reference_points = reference_points[i] * scale_f[None, :]
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0)
        # [batch_size, time * num_queries_per_frame, 2], in image size
        reference_points = new_reference_points

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q)
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride)
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                              locations.reshape(1, 1, 1, h, w, 2)  # [batch_size, time, num_queries_per_frame, h, w, 2]
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3,
                                                      4)  # [batch_size, time, num_queries_per_frame, 2, h, w]

            # concat features
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w',
                                   q=q)  # [batch_size, time, num_queries_per_frame, c, h, w]
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w',
                                   q=q)  # [batch_size, time, num_queries_per_frame, c, h, w]
        mask_features = mask_features.reshape(1, -1, h, w)

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic mask conv
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0])
        mask_logits = mask_logits.reshape(-1, 1, h, w)

        # upsample predicted masks
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(b, num_queries, mask_logits.shape[-2], mask_logits.shape[-1])

        return mask_logits  # [batch_size, time * num_queries_per_frame, h, w]

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


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


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class TrackEmbedding(nn.Module):

    def __init__(self, dim_in, hidden_dim, dim_out, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_out)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_out)

        self.activation = nn.ReLU(True)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)

        return tgt

def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb



def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'custom':
            num_classes = 17
    device = torch.device(args.device)

    # backbone
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:  # always true
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_alpha=args.focal_alpha)
    criterion.to(device)
    ########
    model = TPP(
        args,
        backbone,
        transformer,
        criterion=criterion,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        mask_dim=args.mask_dim,
        dim_feedforward=args.dim_feedforward,
        controller_layers=args.controller_layers,
        dynamic_mask_channels=args.dynamic_mask_channels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        rel_coord=args.rel_coord
    )
    #################

    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors
