import torch
import inspect
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.utils import clear_memory, get_best_device
from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_Marlin,
    WQLinear_GEMVFast,
)
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)
from transformers.models.qwen2.modeling_qwen2 import (Qwen2DecoderLayer, Qwen2ForCausalLM)
from transformers.models.xlm_roberta.modeling_xlm_roberta import (XLMRobertaForSequenceClassification, XLMRobertaLayer)
from typing import List, Tuple, Union

class AwqQuantizer:
    def __init__(
        self,
        awq_model,
        model,
        tokenizer,
        w_bit,
        group_size,
        zero_point,
        version,
        calib_data,
        split,
        text_column,
        duo_scaling,
        modules_to_not_convert=None,
        export_compatible=False,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.export_compatible = export_compatible
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.modules, self.module_kwargs, self.inps, self.attention_mask = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w

    def quantize(self):
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count())
                else:
                    best_device = get_best_device()

                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            #import pdb; pdb.set_trace()
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            #import pdb; pdb.set_trace()
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in tqdm(module_config, desc="Best Scales", leave=False)
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # [STEP 4]: Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)

            clear_memory()

    def pack(self):
        for i in tqdm(range(len(self.modules)), desc="Packing"):
            named_linears = get_named_linears(self.modules[i])
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            self._apply_quant(self.modules[i], named_linears)
            clear_memory()

    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(get_best_device()).half()

            #import pdb; pdb.set_trace()
            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data
            )

            if self.version == "gemm":
                scales = scales.t().contiguous()
                if zeros is not None:
                    zeros = zeros.t().contiguous()
                q_linear_module = WQLinear_GEMM

            elif self.version == "gemv":
                q_linear_module = WQLinear_GEMV

            elif self.version == "marlin":
                q_linear_module = WQLinear_Marlin

            elif self.version == "gemv_fast":
                q_linear_module = WQLinear_GEMVFast

            else:
                raise ValueError(f"Unknown version {self.version}")

            #import pdb; pdb.set_trace()
            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)
            clear_memory()

    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        #import pdb; pdb.set_trace()
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            if attention_mask is not None:
                module_output = module(x, attention_mask, **module_kwargs)
            else:
                module_output = module(x, **module_kwargs)
            #module_output = module(x, attention_mask, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in tqdm(
                partitioned_inputs, desc="Module forward", leave=False
            ):
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        #import pdb; pdb.set_trace()
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.group_size)
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)
        clear_memory(x_sum)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        with tqdm(range(n_grid), desc="Grid Search", leave=False) as pbar:
            for ratio in pbar:
                # create new scales
                ratio = ratio / n_grid

                # NOTE: s^-1 * x is fused here, according to paper
                if self.duo_scaling:
                    scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
                else:
                    scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                scales_view = scales.view(1, -1).to(device)

                # avoid scaling values that overflow
                scales[torch.isinf(scales)] = 1
                scales[torch.isnan(scales)] = 1

                # Q(W * s)
                for fc in linears2scale:
                    fc.weight.mul_(scales_view)
                    fc.weight.data = (
                        self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                    )

                # W * X
                int_w_output = self._module_forward(x, module2inspect, kwargs)

                # compute mean squared error (L2 norm)
                loss = self._compute_loss(fp16_output, int_w_output, device)

                history.append(loss)
                if loss < best_error:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales.clone()
                module2inspect.load_state_dict(org_sd)
                pbar.set_description(f"Grid Search (Best: {best_ratio})")

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        with tqdm(
            zip(fp16_chunks, int_w_chunks),
            total=len(fp16_chunks),
            desc="Computing Loss",
            leave=False,
        ) as pbar:
            for fp16_chunk, int_w_chunk in pbar:
                chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
                loss += chunk_loss
                pbar.set_description(f"Computing Loss (loss: {loss:.2f})")

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in tqdm(named_linears, desc="Computing Best Clip", leave=False):
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    def init_quant(self, n_samples=128, max_seq_len=512):

        #if isinstance(self.model, Qwen2ForCausalLM):
        #    print("self.model is a instance of Qwen2ForCausalLM")
        #elif isinstance(self.model, XLMRobertaForSequenceClassification):
        #    print("self.model is a instance of XLMRobertaForSequenceClassification")

        modules = self.awq_model.get_model_layers(self.model)

        if isinstance(self.model, XLMRobertaForSequenceClassification):
            print("self.model is a instance of XLMRobertaForSequenceClassification")
            #sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]] = \
            #    [("hello world", "nice to meet you"), ("head north", "head south")]
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]] = \
                [["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 981\n\n{'ID': 'r9kvblc0000', '标题': '电影领笑员', '分类': '电影', '标签': '科幻,恐怖,惊悚,悬疑,爱情,传记,冒险,历史,文化,动作,犯罪,科技,战争,奇幻', '看点': '功夫,复仇,暴力', '演员': '范丞丞,沈腾,郭德纲', '导演': '无', '发布年份': '2024.0', '适合年龄段': '19-34岁男', '情绪': 'nan', '天气': '无\\n\\n理由：根据提供的影视信息，“电影领笑员”是一部盘点电影中爆笑名场面的作品，其内容以幽默搞笑为主。而天气列表中的各项天气，如晴天、多云等，通常用来形容氛围与情绪。由于这是一种喜剧类的影视作品，其内容并不直接与任何特定的天气状况相契合，因此，没有适合的天气可以与之对应。时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.749', '简介': '无\\n\\n理由：根据提供的影视信息，“电影领笑员”是一部盘点电影中爆笑名场面的作品，其内容以幽默搞笑为主。而天气列表中的各项天气，如晴天、多云等，通常用来形容氛围与情绪。由于这是一种喜剧类的影视作品，其内容并不直接与任何特定的天气状况相契合，因此，没有适合的天气可以与之对应。时适合观看。盘点电影中各种爆笑名场面,一个比一个搞笑。', '内容源': 'tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 724\n\n{'ID': 'n6leof2k8800', '标题': '第二十条', '分类': '电影', '标签': '院线,喜剧,家庭,搞笑,社会', '看点': '亲情,正能量,社会问题,治愈', '演员': '雷佳音,马丽,赵丽颖', '导演': '张艺谋', '发布年份': '2024.0', '适合年龄段': 'nan', '情绪': '可缓解孤独,失望,愤怒,焦虑情绪。', '天气': '无\\n\\n理由：根据提供的影视作品信息，“第二十条”是一部包含喜剧、家庭、搞笑元素的电影，同时涉及社会问题、亲情、正能量和治愈主题。影片中虽有角色面临的困境和挑战，但整体氛围倾向于通过幽默和积极向上的态度来展现故事。天气列表中的选项多为自然天气现象，与影片的情感基调和内容并不直接对应。因此，没有一种天气能特别契合这部电影的氛围，故返回“无”。时适合观看。', '行业': 'nan', '声纹': '男青年,女青年,男青少年,女青少年,女老年,男老年', 'rating_score': '1.463', '简介': '无\\n\\n理由：根据提供的影视作品信息，“第二十条”是一部包含喜剧、家庭、搞笑元素的电影，同时涉及社会问题、亲情、正能量和治愈主题。影片中虽有角色面临的困境和挑战，但整体氛围倾向于通过幽默和积极向上的态度来展现故事。天气列表中的选项多为自然天气现象，与影片的情感基调和内容并不直接对应。因此，没有一种天气能特别契合这部电影的氛围，故返回“无”。时适合观看。可缓解孤独,失望,愤怒,焦虑情绪。这一年的不容易谁能懂？自打挂职到市检察院，韩明的糟心事就接二连三。儿子韩雨辰打了校领导儿子并拒绝道歉，妻子李茂娟义愤填膺继而揍了校领导，补刀成功；搭档吕玲玲分歧不断，案件久拖不决；又一起案件相关人郝秀萍被逼入绝境，万分危急。情与法的较量在展开，事业与家庭的平衡在进行，韩明决定赌上一切，用自己的方式给公平和正义一个说法……', '内容源': 'youku,tencent,yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3376\n\n{'ID': 'm49v1fnc4400', '标题': '冰雪大围捕', '分类': '电影', '标签': '动作,犯罪,警匪', '看点': '真实事件改编,警察,硬汉,热血', '演员': '樊少皇,释彦能,李智（演员）', '导演': '柯伯龙', '发布年份': '2024.0', '适合年龄段': 'nan', '情绪': '可缓解愤怒,焦虑情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.738', '简介': '下雪时适合观看。可缓解愤怒,焦虑情绪。影片根据真实事件改编。一场当街持枪杀人案掀起轩然大波，刑侦支队长周宇桐和徒弟王峰几经追查锁定职业杀手，实施抓捕之际，雇凶杀人的范家两兄弟不惜铤而走险，设计大爆炸来杀人灭口，王峰在行动中不幸牺牲，凶残嫌犯却杀人劫车，试图穿越林海雪原出境。来不及悲痛的周宇桐怒火追凶，身受重伤却坚持咬紧不放，在零下30度的极端严寒中与悍匪范巨友展开了殊死搏斗……', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 159\n\n{'ID': 'crf2oa1scc00', '标题': '我要我们在一起', '分类': '电影', '标签': '爱情,青春,恋爱', '看点': '人生,成长', '演员': '屈楚萧,张婧仪,孙宁', '导演': '沙漠', '发布年份': '2021.0', '适合年龄段': '19-34岁女', '情绪': '可缓解孤独,失望情绪。', '天气': '阴天,雨天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.777', '简介': '阴天,雨天时适合观看。可缓解孤独,失望情绪。你有没有爱过一个人，曾经拼了命，只为和TA在一起。十年前，差生吕钦扬当众告白凌一尧，两人从校园步入社会，为了让她幸福，他不惜以命相搏。然而金钱、房子、婚姻等现实的考验，却将两人越推越远。十年长跑，他们能否还记得曾经刻在心底的约定：我要我们在一起。', '内容源': 'tencent,youku,yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 577\n\n{'ID': '4p6p0lvk0000', '标题': '冰雪奇缘', '分类': '电影', '标签': '动画,喜剧,冒险,奇幻,歌舞,青春,伦理,音乐', '看点': '亲情,巨制,口碑佳片,迪士尼,警察,成长,野外生存,艾莎,安娜,萝莉', '演员': '克里斯汀·贝尔,伊迪娜·门泽尔,乔纳森·格罗夫', '导演': '詹妮弗·李', '发布年份': '2013.0', '适合年龄段': '19-34岁男', '情绪': '可缓解孤独,失望,沮丧情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.788', '简介': '下雪时适合观看。可缓解孤独,失望,沮丧情绪。在四面环海、风景如画的阿伦达王国，生活着两位可爱美丽的小公主，艾莎（伊迪娜·门泽尔 配音）和安娜（克里斯汀·贝尔 配音）。艾莎天生具有制造冰雪的能力，随着年龄的增长，她的能力越来越强，甚至险些夺走妹妹的生命。为此国王紧闭宫门，也中断了两个女儿之间的联系。悲哀的海难过后，艾莎终于到了加冕的年龄，各国王宫纷纷前来祝贺。艾莎战战兢兢，唯恐被人识破隐藏了多年的秘密。然而当听说安娜将要和初次见面的南群岛王子汉斯（圣蒂诺·方塔纳 配音）结婚时，依然情绪失控露出了马脚。在此之后她逃到山中，构建了属于自己的冰雪王国，而阿伦达也陷入可怕的寒冷之中。安娜独自来到山中，在拉冰青年克里斯托弗（乔纳森·格罗夫 配音）的帮助下总算来到姐姐的宫殿，她能否让国家重新找回失落的绿意？', '内容源': 'youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3372\n\n{'ID': 'goe3ek6o4400', '标题': '汤米当医生', '分类': 'Education', '标签': 'nan', '看点': 'nan', '演员': 'nan', '导演': 'nan', '发布年份': '2022.0', '适合年龄段': '4-6岁女', '情绪': 'nan', '天气': 'nan', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.738', '简介': '小猫汤米穿上白大褂，戴上听诊器，给大家看病来了！小鸭子捂着肚子喊疼，因为他吃饭前不洗手导致饮食不卫生；小鳄鱼流着眼泪，因为他总看电视不休息。小朋友，你想一想，平时有没有什么不良习惯？有的话，及时改正哦！我们要健健康康长大！', '内容源': 'tencent,yinhe,youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3742\n\n{'ID': 'r9pqfh00000', '标题': '冰雪奇缘（普通话）', '分类': '电影', '标签': '动画,冒险,喜剧,奇幻,歌舞,青春,伦理', '看点': '巨制,亲情,口碑佳片,野外生存,成长,艾莎,好莱坞,安娜,萝莉', '演员': '克里斯汀·贝尔,伊迪娜·门泽尔,乔纳森·格罗夫', '导演': '克里斯·巴克', '发布年份': '2014.0', '适合年龄段': '19-34岁女', '情绪': '可缓解孤独,失望,沮丧情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.741', '简介': '下雪时适合观看。可缓解孤独,失望,沮丧情绪。在四面环海、风景如画的阿伦黛尔王国，生活着两位可爱美丽的小公主，艾莎和安娜。艾莎天生具有制造冰雪的能力，随着年龄的增长，她的能力越来越强，甚至险些夺走妹妹的生命。为此国王紧闭宫门，也中断了两姐妹的联系。悲哀的海难过后，艾莎终于到了加冕的年龄，各国王公齐来祝贺。艾莎战战兢兢，唯恐被人识破隐藏了多年的秘密。然而当听说安娜将要和初次见面的南埃尔斯王子汉斯结婚时，依然情绪失控露出了马脚。在此之后她逃到山中，构建了属于自己的冰雪王国，而阿伦黛尔也陷入可怕的寒冷之中。', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1515\n\n{'ID': 'r9etg3k0000', '标题': '末日时在做什么？有没有空？可以来拯救吗？', '分类': '动漫', '标签': '奇幻,恋爱,动画,动作', '看点': '热血,催泪,少女', '演员': '新井良平,田所梓,齐藤真知子', '导演': '和田纯一', '发布年份': '2016.0', '适合年龄段': '19-34岁男', '情绪': '可缓解孤独,失望,悲伤情绪。', '天气': '阴天,雨天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.597', '简介': '阴天,雨天时适合观看。可缓解孤独,失望,悲伤情绪。「人类」遭到非比寻常的「兽」的蹂躏而灭亡了。除了独自从数百年沉眠中苏醒的青年威廉以外。唯有「圣剑」与使用圣剑的妖精兵能代替「人类」打倒「兽」。战斗过后，「圣剑」能再次被他人使用，但用尽力量的妖精兵们却会殒命。「至少，我也希望自己不用消失，也想让别人记住。我也想留下羁绊啊。」这是注定赴死的妖精少女们和青年教官共度的，既短暂又灿烂的日子。', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 237\n\n{'ID': 'r9fjcm00000', '标题': '冰雪女王', '分类': '电影', '标签': '动作,动画,奇幻,冒险,儿童,童话', '看点': 'nan', '演员': '布里吉特·芳达,切尔西·汉德勒,RobertWisden', '导演': '胡大为', '发布年份': '2002.0', '适合年龄段': '19-34岁男', '情绪': 'nan', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.27', '简介': '下雪时适合观看。一场意外让恶魔之镜碎裂散落人间，其中的一小片不幸落入了小男孩凯（Jeremy Guilbaut 饰）的眼中，因为碎片的魔力，凯的个性开始变得古怪起来，世间的一切在他眼中都丑陋无比，他也因此远离了朋友和家人，封闭起了自己的内心。某日，冰雪王后（布里吉特·芳达 Bridget Fonda 饰）出现了，她将离群索居的凯带回了自己的国度。格尔达（切尔西·汉德勒 Chelsea Hobbs 饰）是凯的青梅竹马，两人从小玩到大，彼此之间感情十分要好。对于凯的下落不明，格尔达感到十分焦急。当得知凯来身处冰雪王后的宫殿之时，格尔达毅然决然地踏上了充满危险的旅程。最终，她能否顺利抵达目的地，救回被冰雪王后迷惑了心智的凯呢？', '内容源': 'tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 2794\n\n{'ID': 'r9ea5r80000', '标题': '冰雪奇缘', '分类': '电影', '标签': '动画,喜剧,冒险,歌舞,奇幻,伦理,动作,青春,音乐,滑冰', '看点': '成长,亲情,巨制,迪士尼,萝莉,艾莎,警察,口碑佳片,安娜,好莱坞,野外生存,滑冰', '演员': '克里斯汀·贝尔,伊迪娜·门泽尔,乔纳森·格罗夫', '导演': '克里斯·巴克', '发布年份': '2014.0', '适合年龄段': '19-34岁男', '情绪': '温馨治愈，驱散恐惧。', '天气': '下雪,下雪,冰雹时适合观看。', '行业': '滑冰', '声纹': 'nan', 'rating_score': '0.762', '简介': '包含滑冰相关元素。下雪,下雪,冰雹时适合观看。温馨治愈，驱散恐惧。在四面环海、风景如画的阿伦黛尔王国，生活着两位可爱美丽的小公主，艾莎和安娜。艾莎天生具有制造冰雪的能力，随着年龄的增长，她的能力越来越强，甚至险些夺走妹妹的生命。为此国王紧闭宫门，也中断了两姐妹的联系。悲哀的海难过后，艾莎终于到了加冕的年龄，各国王公齐来祝贺。艾莎战战兢兢，唯恐被人识破隐藏了多年的秘密。然而当听说安娜将要和初次见面的南埃尔斯王子汉斯结婚时，依然情绪失控露出了马脚。在此之后她逃到山中，构建了属于自己的冰雪王国，而阿伦黛尔也陷入可怕的寒冷之中。安娜独自来到山中，在山民克斯托夫的帮助下总算来到姐姐的宫殿，她能否让国家重新找回失落的绿意？', '内容源': 'tencent,yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3760\n\n{'ID': 'kmt5fh7c4400', '标题': '造梦之家', '分类': '电影', '标签': '传记', '看点': 'nan', '演员': '米歇尔·威廉姆斯,保罗·达诺,塞斯·罗根', '导演': '史蒂文·斯皮尔伯格', '发布年份': '2022.0', '适合年龄段': '19-34岁女', '情绪': 'nan', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.674', '简介': '晴天,多云时适合观看。本片是斯皮尔伯格受成长经历启发的初心之作，讲述了主人公萨米·法贝尔曼（加布里埃尔·拉贝尔 饰）的成长经历。萨米从小就爱上了电影，并尝试创作属于自己的电影。这一兴趣得到了他的艺术家母亲米茨（米歇尔·威廉姆斯 饰）、计算机工程师父亲伯特（保罗·达诺 饰）以及家中其他人的一致支持。如果说电影是造梦的艺术，那么法贝尔曼一家就是一个“造梦之家”。多年之后，萨米已成长为一个天才的少年导演，凭热爱创作出一部部令人惊喜的业余电影。但意外的是，通过摄影机的镜头，他发现了一个关于母亲的心碎真相。而这将改变他与整个家庭的未来。', '内容源': 'youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 361\n\n{'ID': '820mp73s0400', '标题': '冰雪奇缘2', '分类': '电影', '标签': '动画,冒险,歌舞,喜剧,奇幻', '看点': '迪士尼,安娜,高票房,艾莎,冰雪奇缘', '演员': '克里斯汀·贝尔,伊迪娜·门泽尔,乔纳森·格罗夫', '导演': '克里斯·巴克', '发布年份': '2019.0', '适合年龄段': '19-34岁男', '情绪': '可缓解孤独,失望,沮丧,焦虑情绪。', '天气': '下雪,雾天,冰雹时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.862', '简介': '下雪,雾天,冰雹时适合观看。可缓解孤独,失望,沮丧,焦虑情绪。历经严酷考验，阿伦戴尔王国终于回归往日平静。艾莎女王、安娜公主以及他们的好友雪宝、克里斯托弗、驯鹿斯文过着平静安逸的生活。可是最近一段时间，艾莎总会被一段神秘的吟唱所困扰，为了追寻真相，她义无反顾踏上了征途。担心姐姐的安全，安娜和雪宝、克里斯托弗他们紧紧跟随。在那座常年被浓雾所笼罩的森林里，不仅藏着神秘的自然力量，更隐藏着关于阿伦戴尔王国、艾莎的魔法来源以及两位公主父母丧生等一系列的秘密。艾莎开启了一段寻找自我的旅程……', '内容源': 'youku,tencent,yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 914\n\n{'ID': 'dicu87bs8800', '标题': '龙虎武师', '分类': '电影', '标签': '动作', '看点': '功夫,硬汉,热血', '演员': '徐克,袁和平,洪金宝', '导演': '魏君子', '发布年份': '2021.0', '适合年龄段': '35-54岁男', '情绪': 'nan', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.722', '简介': '晴天,多云时适合观看。纪录电影《龙虎武师》集结洪金宝、袁和平、程小东、元华、甄子丹、钱嘉乐等华语电影最强“武师阵容”，首次全方位揭秘香港龙虎武师长达六十余年风云变幻，展现中国功夫影人搏命人生路。', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3396\n\n{'ID': '9g1rjhe00400', '标题': '雪中悍刀行', '分类': '电视剧', '标签': '武侠,古装,权谋', '看点': '成长,爆款剧,小说改编,爽剧', '演员': '张若昀,李庚希,胡军', '导演': '宋晓飞', '发布年份': '2021.0', '适合年龄段': '19-34岁男', '情绪': '可缓解孤独,失望,沮丧情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.917', '简介': '下雪时适合观看。可缓解孤独,失望,沮丧情绪。为逃避做隋珠公主的驸马，“天下第一纨绔”的北椋世子徐凤年在父亲徐骁的安排下褪去锦衣华服，初进江湖，和马夫老黄苦中作乐，结识了众多江湖人士。三年游历归来，韬光养晦的徐凤年洗去浮尘，始终不想按照老爹铺好的人生轨道走，更不愿接手北椋，因为成为北椋王，就意味着要成为一个没有感情的孤家寡人。但当徐凤年雪中冠礼，得知一个个至亲离他而去，为他牺牲，经历人生的至暗时刻后，终于下定决心，要当一个和父亲完全不一样的北椋王，再难也不能妥协，遂苦学武艺，凭借赤子之心和勤学苦练，成为武者，而后率丫鬟姜泥、剑仙李淳罡等护卫，二进江湖，用悍刀闹得武林势力鸡飞狗跳，看似按老爹的套路下棋，实则踏雪独闯，力抗命运安排，渐渐培植了愿为自己效忠的武当、江南文坛、西楚、徽山轩辕等武林新势力，也通过种种线索发现母亲吴素之死的真相。漫天飞雪，徐凤年一人一刀一腔仇，用自己的身躯扛起北椋战旗，最终长成为北椋王合格的接班人。', '内容源': 'tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 628\n\n{'ID': 'mrfhgg8g8800', '标题': '人间世', '分类': '电影', '标签': '家庭,院线', '看点': '亲情,治愈系', '演员': '王思蓉', '导演': '陶涛（导演）', '发布年份': '2022.0', '适合年龄段': 'nan', '情绪': '可缓解悲伤,孤独,失望,焦虑情绪。', '天气': '阴天,雨天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.682', '简介': '阴天,雨天时适合观看。可缓解悲伤,孤独,失望,焦虑情绪。电影《人间世》由陶涛、张琪、秦博、范士广四位导演共同执导，陶涛，张琪联合监制，秦博、范士广任总策划。影片选取两位身患绝症的病人，将目光聚焦于她们的家庭，记录下她们人生最后的时光，书写了一首关乎爱的生命诗篇。电影《人间世》是一部不同于电视版的大银幕作品，秉承对生命的敬畏，以全新的主题立意，构建不一样的叙事视角与故事结构，将电影蒙太奇手法创造性融入真实感人的纪录影像，向电影中这些真实，可爱并飞扬着生命力量的人物致以最大的敬意。人间世，爱是感同身受的勇气，触手可及！', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 2163\n\n{'ID': 'k8bkunsscc00', '标题': '冰雪狙击2', '分类': '电影', '标签': '战争,动作,历史,军事,抗战', '看点': '主旋律,正能量,励志,热血', '演员': '刘晓庆,王新军,于荣光', '导演': '赵锐勇', '发布年份': '2023.0', '适合年龄段': '35-54岁男', '情绪': '可缓解愤怒,焦虑,失望情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.683', '简介': '下雪时适合观看。可缓解愤怒,焦虑,失望情绪。1947年东北，国民党军抢夺胜利果实，攻击解放军根据地。战事胶着，百姓罹难。解放军政委吴润东派遣侦察排长、狙击手杨建峰上猛虎山求援张东山。张东山下山调停，可国民党曾天养却勾结悍匪许天雷，在下山路上假扮解放军击杀张东山，丹娘誓要报杀父之仇，杨建峰智勇揭开国民党的阴谋，并对丹娘动之以情、晓之以理，最终，丹娘弃暗投明，和杨建峰一起血战猛虎山，将九山十八岗悍匪一网打尽。', '内容源': 'yinhe,youku,tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 2229\n\n{'ID': 'k573oa9s4400', '标题': '“湾区升明月”2023大湾区电影音乐晚会', '分类': '综艺', '标签': '真人秀', '看点': 'nan', '演员': '蓝羽,吴京,刘乃奇', '导演': '王平久', '发布年份': '2023.0', '适合年龄段': '19-34岁女', '情绪': 'nan', '天气': '晴天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.76', '简介': '晴天时适合观看。2021年中秋月圆夜，“湾区升明月”大湾区中秋电影音乐晚会破圈突围、全网霸屏、刷爆热搜，群星璀璨汇聚湾区，点燃心中金曲回忆。全明星、全金曲、全爆款、全共鸣，以电影与音乐为媒，共享电影之美、和合之美，共同感受大湾区文化氛围。 时隔两年，歌会重启；经典旋律，意犹未尽；湾区之约，香港再续。为响应建设粤港澳大湾区的重大国家战略，正值香港回归祖国26周年之际，“湾区升明月”2023大湾区电影音乐晚会渡江而来，落户香港，全新升级，再度唱响，以更高视野和格局，面向中国和世界，打造粤港澳大湾区闪闪发光的经典文化品牌。百余位来自海峡两岸暨香港、澳门地区的电影人、音乐人，将在香江之畔热忱集结，书写属于粤港澳大湾区的中国式现代化新图景，为香港回归祖国26周年献礼。', '内容源': 'tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 431\n\n{'ID': '820mpakg0400', '标题': '冰雪奇缘2（普通话）', '分类': '电影', '标签': '动画,冒险,歌舞,喜剧,奇幻,译制片', '看点': '迪士尼,安娜,艾莎', '演员': '克里斯汀·贝尔,伊迪娜·门泽尔,乔纳森·格罗夫', '导演': '珍妮弗·李', '发布年份': '2019.0', '适合年龄段': '19-34岁男', '情绪': 'nan', '天气': '雾天,下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.767', '简介': '雾天,下雪时适合观看。历经严酷考验，阿伦戴尔王国终于回归往日平静。艾莎女王、安娜公主以及他们的好友雪宝、克里斯托弗、驯鹿斯文过着平静安逸的生活。可是最近一段时间，艾莎总会被一段神秘的吟唱所困扰，为了追寻真相，她义无反顾踏上了征途。担心姐姐的安全，安娜和雪宝、克里斯托弗他们紧紧跟随。在那座常年被浓雾所笼罩的森林里，不仅藏着神秘的自然力量，更隐藏着关于阿伦戴尔王国、艾莎的魔法来源以及两位公主父母丧生等一系列的秘密。艾莎开启了一段寻找自我的旅程……', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3380\n\n{'ID': 'mtdb324occ00', '标题': '永安镇故事集', '分类': '电影', '标签': '社会,搞笑,院线,喜剧', '看点': '文艺,无厘头,女性,人生', '演员': '杨子姗,黄米依,刘洋', '导演': '魏书钧', '发布年份': '2023.0', '适合年龄段': 'nan', '情绪': '可缓解孤独,失望,沮丧情绪。', '天气': '多云,阴天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.67', '简介': '多云,阴天时适合观看。可缓解孤独,失望,沮丧情绪。电影《永安镇故事集》是一部“关于电影”的电影，叙述了一个剧组入住拍摄地，给这个原本宁静的小镇带来一丝波澜，然而波澜过后，一切又重归宁静的故事。影片分为三个章节——独自等待、看上去很美、冥王星时刻，小镇和剧组串联起三个章节的故事和其中的人物——与小镇格格不入的饭店老板娘、回不去故乡的女明星、陷入创作分歧的导演与编剧……因为剧组的到来，每一个置身这座小镇的人心中都荡起一丝涟漪，但涟漪终会散去，一切也终将重归平静。', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1869\n\n{'ID': '8h2dliss0400', '标题': '少年歌行 风花雪月篇', '分类': '动漫', '标签': '动画,古风,国漫,武侠', '看点': '热血,励志', '演员': '张博恒,张杰,杨婧', '导演': '陈升垚', '发布年份': '2021.0', '适合年龄段': '13-18岁男', '情绪': 'nan', '天气': '雪天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.767', '简介': '雪天时适合观看。第一季，忘忧大师圆寂后，江湖中人争夺忘忧的弟子——有着特殊身份的无心。而无心却一心只想护送忘忧大师的遗骨落叶归根。在雷无桀和萧瑟的帮助下，三人终于抵达忘忧的故乡并安葬了他，还帮助无心摆脱追杀，回到自己的家乡天外天。本季，无心离开后，萧瑟和雷无桀来到雪月城，三城主司空长风认出萧瑟身份，强行收其为徒。雷无桀闯登天阁求见二城主雪月剑仙李寒衣，想让李寒衣去雷门看望自己的师父雷轰。二人定下三剑之约。数月后，习武有成的雷无桀成功接下李寒衣的三剑，李寒衣赴约下山。随着萧瑟进入雪月城，天启的其他王子也蠢蠢欲动、纷纷寻求盟友。七皇子赤王萧羽不远万里去天外天寻求无心的支持。白王萧崇不仅勾结暗河暗杀萧瑟，还联合无双城为自己的后盾。一场针对整个雪月城的阴谋正在酝酿。', '内容源': 'youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1524\n\n{'ID': 'k5g0nnuc8800', '标题': '“湾区升明月”2023大湾区电影音乐晚会', '分类': '综艺', '标签': '晚会,音乐', '看点': 'nan', '演员': '蓝羽,吴京,刘乃奇', '导演': '王平久', '发布年份': '2023.0', '适合年龄段': '19-34岁女', '情绪': 'nan', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.743', '简介': '晴天,多云时适合观看。湾区谱新篇，声影耀香江。6月29日，“湾区升明月”2023大湾区电影音乐晚会将于中国香港精彩唱响。百余位来自海峡两岸暨港澳地区的电影人、音乐人、文体科技界代表汇聚香江之畔，穿越光影与音乐的长河，用温暖的情谊与热烈的奋斗，共同抒写属于粤港澳大湾区的中国式现代化新图景。', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3412\n\n{'ID': 'r9kvl6g0000', '标题': '熊出没之雪岭熊风', '分类': '电影', '标签': '动画,喜剧,冒险,搞笑', '看点': '熊出没', '演员': '张伟,张秉君,谭笑', '导演': '丁亮', '发布年份': '2015.0', '适合年龄段': '4-6岁男', '情绪': '可缓解孤独,失望,愤怒,焦虑,震惊,怀疑,羞愧,自责,悔恨,尴尬情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.711', '简介': '下雪时适合观看。可缓解孤独,失望,愤怒,焦虑,震惊,怀疑,羞愧,自责,悔恨,尴尬情绪。在狗熊岭百年不遇的大雪中，熊二偶遇了小时候曾有过一面之缘的神秘小伙伴，除了重逢的喜悦，小伙伴也给熊二带来了不少麻烦：穷凶极恶的追猎者、神秘而未知的重大传说。一系列的阴差阳错，熊大熊二光头强和动物们不可避免地被卷入其中。在小镇和森林中，他们闹出了不少惊险又好笑的意外，在传说的驱使下，一行人踏上了前往白熊山的旅程，一路上，他们经历了欢笑和感动，勇气日渐增长，友谊也越加深厚，熊大和熊二学会了理解对方，矛盾也渐渐地化解。可是，一场灾难意外地爆发，在千钧一发之际，熊二鼓起勇气，承担起了拯救大家的责任，危机最终圆满解决。', '内容源': 'tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1381\n\n{'ID': '644sfi180400', '标题': '一切如你', '分类': '电影', '标签': 'nan', '看点': '公益,社会问题,亲情,正能量,老戏骨,治愈', '演员': '张勇手,管宗祥,刘江', '导演': '黄宏', '发布年份': '2019.0', '适合年龄段': '19-34岁女', '情绪': '可缓解孤独,失望情绪。', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.694', '简介': '晴天,多云时适合观看。可缓解孤独,失望情绪。影片的初衷是，有四位年轻人，他们用镜头记录下了爷爷奶奶的现实生活，在创作过程中，他们了解到中国当今的老龄化现状，有太多的老人缺少陪伴，于是，他们决定用自己的所学，拍摄这部电影《一切如你》。从年轻人角度出发，讲述了十个“养老、孝老、敬老”的故事。他们的举动得到了老一辈电影艺术家的鼎力支持，这次拍摄是一次难忘的创作历程，更是一次肩负使命的文化传承。', '内容源': 'youku,yinhe,tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3811\n\n{'ID': 'isv38vescc00', '标题': '在暴雪时分', '分类': '电视剧', '标签': '爱情,都市', '看点': 'nan', '演员': '吴磊,赵今麦,王星越', '导演': '黄天仁', '发布年份': '2024.0', '适合年龄段': 'nan', '情绪': '可缓解孤独,失望,沮丧情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.905', '简介': '下雪时适合观看。可缓解孤独,失望,沮丧情绪。昔日天才级职业斯诺克选手林亦扬与当红九球选手殷果相遇在一场十年难得一遇的暴雪之夜。在殷果表弟的助攻之下，两人慢慢熟识，殷果也了解到林亦扬的往事。原来林亦扬曾经也是轰动一时的斯诺克天才，在一次比赛中因对裁判判罚有异议，争取时无意冲撞裁判而被禁赛，少年气盛的他不服判决，竟直接宣布退役，从此离开了自己引以为傲的斯诺克事业。殷果的出现，让林亦扬的人生轨迹再次转变，林亦扬不仅开始努力地追求殷果，还在殷果的带动下重拾最初的梦想，返回斯诺克赛场。最终二人共攀事业高峰，一同为职业台球事业贡献力量，并携手教练与队友，共推全民健身台球项目发展。', '内容源': 'tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 637\n\n{'ID': 'a3186bkg0400', '标题': '刺猬索尼克（普通话）', '分类': '电影', '标签': '动画,动作,喜剧,冒险,科幻,奇幻,搞笑', '看点': '超自然,成长,游戏改编,励志,穿越', '演员': '詹姆斯·麦斯登,本·施瓦兹,提卡·森普特', '导演': '杰夫·福勒', '发布年份': '2020.0', '适合年龄段': '13-18岁男', '情绪': '可缓解孤独,失望,愤怒,焦虑,震惊,怀疑,羞愧,自责,悔恨,尴尬情绪。', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.682', '简介': '晴天,多云时适合观看。可缓解孤独,失望,愤怒,焦虑,震惊,怀疑,羞愧,自责,悔恨,尴尬情绪。电影讲述了拥有音速奔跑能力的刺猬索尼克在地球上展开新生活的故事。索尼克和他的新朋友汤姆（詹姆斯·麦斯登 饰）联手保护地球，阻止邪恶的蛋头博士（金·凯瑞 饰）统治世界的计划。这部适合全家观看的电影还邀请了迪卡·桑普特参演，本·施瓦茨为索尼克配音。', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1170\n\n{'ID': 'eea671uc4400', '标题': '雪鹰领主', '分类': '电视剧', '标签': '爱情,古装,科幻,奇幻', '看点': 'nan', '演员': '许凯,古力娜扎,白澍', '导演': '李达超', '发布年份': '2023.0', '适合年龄段': '19-34岁男', '情绪': '可缓解孤独,失望,沮丧情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.762', '简介': '下雪时适合观看。可缓解孤独,失望,沮丧情绪。拥有太古血脉的少年东伯雪鹰，自幼丧父、与弟弟相依为命，为救母卷入正魔争斗。在与魔族云谲波诡的对抗中，雪鹰英勇无惧，屡入险境。在夏族危难之际，他初心不易，百折不屈，不断突破自我极限。其间，雪鹰与神族后裔女子余靖秋相识相恋，两人历经磨难，不畏艰险，成为知心爱人，生死与共。共同保卫夏族，守护家园，展现了夏族少年风采，最终粉碎魔族阴谋，捍卫疆土完整，书写了一曲回肠荡气的热血神话。本剧旨在告诉年轻一代观众，立足于民族，立足于大爱，不忘自己的初心，方能成为真正的英雄。', '内容源': 'tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 736\n\n{'ID': 'gpn9o3vc8800', '标题': '误杀瞒天记', '分类': '电影', '标签': '悬疑,犯罪,惊悚', '看点': '警察,口碑佳片,热血', '演员': '阿贾耶·德乌干,塔布,施芮娅·萨兰', '导演': '尼西卡特·卡马特', '发布年份': '2022.0', '适合年龄段': '19-34岁男', '情绪': 'nan', '天气': '阴天,雨天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.732', '简介': '阴天,雨天时适合观看。该片讲述父亲维杰为了维护错手杀死不速之客的家人，用电影里学来的反侦察手法和警察斗智斗勇的故事。', '内容源': 'tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1837\n\n{'ID': 'r917o4k0000', '标题': '雪豹坚强岁月', '分类': '电视剧', '标签': '战争,军旅,冒险,抗战,爱情,年代,军事,人文', '看点': '革命,初恋,硬汉,浙江卫视,热血,励志', '演员': '张若昀,高洋,毛晓彤', '导演': '张健', '发布年份': '2014.0', '适合年龄段': '35-54岁男', '情绪': '可缓解愤怒,失望,悲伤情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.881', '简介': '下雪时适合观看。可缓解愤怒,失望,悲伤情绪。周文为解救遭调戏的女友，杀死日本浪人，闯下大祸。在他的父兄多方解救下，侥幸逃脱并化名周卫国考入军校，由于成绩优异被送到德国留学。抗日战争全面爆发后，周卫国虽然奋勇杀敌，无奈国军接连败退，不断丢城失地，尤其是他的未婚妻为免遭日军凌辱而自杀，自己眼睁睁地看着却无法解救，更让周卫国国仇家恨集于一身。国军溃败，周卫国流落到八路军根据地，他历经波折，从一个只知道单纯杀鬼子报仇发展到成为成熟的八路军指战员，并组建了代号“雪豹”的特战队，为抗战事业做出了贡献。在荣誉和名利面前，他淡然处之，成就了一段传奇。', '内容源': 'tencent,yinhe,youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 2036\n\n{'ID': 'dkl0men84400', '标题': '“湾区升明月”2021大湾区中秋电影音乐晚会', '分类': '综艺', '标签': '晚会,文化,音乐', '看点': 'nan', '演员': '邓超,谢霆锋,蓝羽', '导演': '王平久', '发布年份': '2021.0', '适合年龄段': '19-34岁女', '情绪': 'nan', '天气': '晴天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.698', '简介': '晴天时适合观看。为了进一步推动大湾区经济与文化交流，首届“粤港澳大湾区购物节”活动于2021年9月2日至22日举办，“湾区升明月”2021大湾区中秋电影音乐晚会将于9月21日举办。9月21日中秋节之际，特别推出首届“粤港澳大湾区购物节”群星接力公益直播暨“湾区升明月”大湾区中秋电影音乐晚会融媒体直播活动，让“购物节”的经济交融与“电影音乐晚会”的文化交流交相辉映，以群星文艺表演、知名主播直播带货、访谈互动等形式充分展现大湾区融合发展成果，分享中秋故事、传递爱国心声、讲述大湾区文化故事，献以诚挚的中秋祝福。', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1840\n\n{'ID': '4fgth51c0000', '标题': '雪·葬', '分类': '电影', '标签': 'nan', '看点': '社会问题,政治,改革开放', '演员': '任帅,徐永革,叶鹏', '导演': '高成岗', '发布年份': '2018.0', '适合年龄段': '35-54岁男', '情绪': 'nan', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.701', '简介': '下雪时适合观看。文革中，农村青年赵天佑，因家庭出身为地主，差点沦落为神经病，无奈中入赘山区柳沟河。改革开放后，当选为村委会主任。在爷爷帮助过的长工儿子的帮助下，做起了中药材生意，几年功夫，柳沟河人靠种植中药材发家致富，赵天佑也因此成为农民企业家。常务副市长王华林看到了其中的发展前景，把文阳县开发区升级为云水市中药材开发区，极力主张中药材深加工。王华林后来被冒牌港商茹丽华拉下水，帮助她在云水市玩“空手道”套取资金。农民企业家赵天佑中了茹丽华的美人计，茹丽华骗走了柳沟河农民的中药材，然后潜逃。赵天佑迫于道德压力与法律追问，大雪天跳崖自杀。', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3618\n\n{'ID': 'hr9ni7ao8800', '标题': '小黄人大眼萌：神偷奶爸前传', '分类': '电影', '标签': '动画,喜剧,冒险,科幻,动作,犯罪,都市', '看点': '萌,治愈', '演员': '史蒂夫·卡瑞尔,皮艾尔·柯芬,塔拉吉·P·汉森', '导演': '凯尔·巴尔达', '发布年份': '2022.0', '适合年龄段': '7-12岁男', '情绪': '可缓解孤独,失望,愤怒,焦虑,震惊,怀疑,羞愧,自责,悔恨,尴尬情绪。', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.815', '简介': '晴天,多云时适合观看。可缓解孤独,失望,愤怒,焦虑,震惊,怀疑,羞愧,自责,悔恨,尴尬情绪。影片是2015年推出的小黄人独立电影《小黄人大眼萌》的直接前传。作为《神偷奶爸》的衍生作品，讲述了小黄人们在“前格鲁”时代为其他主人服务的经历，不过少年格鲁曾经出现在这部衍生电影里，续集将围绕他与小黄人们结缘的过程展开。', '内容源': 'tencent,yinhe,youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3773\n\n{'ID': 'h0gaqinkcc00', '标题': '冰冻星球 第一季', '分类': '纪录片', '标签': '自然', '看点': 'nan', '演员': '大卫·爱登堡,李易', '导演': '艾雷斯泰·法瑟吉尔', '发布年份': '2011.0', '适合年龄段': '13-18岁男', '情绪': 'nan', '天气': '下雪,大风,冰雹时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.808', '简介': '下雪,大风,冰雹时适合观看。英国BBC电视台耗时5年制作的纪录片《冰冻地球》即将于近期推出。这部耗资巨大的纪录片用镜头真实的展现了正在逐渐溶解的地球两极，以及生活在这里的各种生物，片中种种景象让人叹为观止。纪录片的解说员大卫·艾登堡（Davi Attenborough）爵士称，这可能是人类在地球气候产生剧烈变化前欣赏到这一景象的最后的机会了。现年85岁的艾登堡爵士说，“这部纪录片捕捉了此前从未记录下来的很多行为和现象。随着时间的流逝，这些影像将会变得越来越珍贵，因为这很可能是我们最后的机会去记录下这些珍贵的场景。虽然在我们到达前的数百年甚至几千年前，地球两极的景象非常壮观，但是最近一个世纪以来，很多变化已经超过了人们的认识。”', '内容源': 'youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1032\n\n{'ID': 'l3fcsb8gcc00', '标题': '碟中谍7：致命清算（上）', '分类': '电影', '标签': '动作,惊悚,冒险', '看点': '特工', '演员': '汤姆·克鲁斯,海莉·阿特维尔,文·瑞姆斯', '导演': '克里斯托夫·迈考利', '发布年份': '2023.0', '适合年龄段': '35-54岁男', '情绪': 'nan', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': '男青少年,女青少年,男青年', 'rating_score': '0.999', '简介': '晴天,多云时适合观看。《碟中谍7》是《碟中谍》系列的第七部，是由克里斯托夫·迈考利执导，汤姆·克鲁斯主演的动作电影，将于2021年11月19日在北美上映。', '内容源': 'youku,tencent,yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3059\n\n{'ID': 'gl6u3tv0cc00', '标题': '雪山飞狐之塞北宝藏', '分类': '电影', '标签': '动作,武侠,爱情,悬疑,冒险,古装', '看点': '复仇,热血', '演员': '赵华为,陈雨锶,吕良伟', '导演': '乔磊', '发布年份': '2022.0', '适合年龄段': '19-34岁男', '情绪': 'nan', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.739', '简介': '下雪时适合观看。十年前，恶人们为抢夺闯王留下的神秘宝藏，设计引出大侠苗人凤与宝藏守护者胡一刀决战。胡苗二人惨死，然而藏宝图却不知去向。十年时间里，他们从未放弃过宝藏的探寻。终于，藏宝图重现江湖，八恶人聚首，一同向飞狐山出发...', '内容源': 'tencent,yinhe,youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 2594\n\n{'ID': 'mvb8rpog4400', '标题': '完美的日子', '分类': '电影', '标签': 'nan', '看点': '人生,文艺', '演员': '役所广司,柄本时生,中野有纱', '导演': '维姆·文德斯', '发布年份': '2023.0', '适合年龄段': 'nan', '情绪': '可缓解孤独,失望情绪。', '天气': '多云,阴天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.722', '简介': '多云,阴天时适合观看。可缓解孤独,失望情绪。这是一部讲述平凡人在日常生活中寻找美好的电影。主人公平山是一名清洁工，对于平凡而规律的工作生活感到满足。除此之外，他热爱音乐、书籍和拍摄树木的照片。一次意外的相遇渐渐揭示了他的过去。', '内容源': 'tencent,youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 941\n\n{'ID': 'mf5im3n44401', '标题': '一闪一闪亮星星', '分类': '电影', '标签': '爱情,奇幻,魔幻,青春,院线', '看点': '成长,暗恋,纯爱,文艺,浪漫,催泪,穿越', '演员': '屈楚萧,张佳宁,傅菁', '导演': '陈小明', '发布年份': '2023.0', '适合年龄段': 'nan', '情绪': '可缓解悲伤,沮丧,孤独,失望情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.722', '简介': '下雪时适合观看。可缓解悲伤,沮丧,孤独,失望情绪。现象级爆款剧集同名电影《一闪一闪亮星星》，原班人马再续纯爱故事，令无数观众翘首以盼的纯爱时空再启，奔赴甜虐暗恋！张万森（屈楚萧 饰）计划在高考后向暗恋已久的女生林北星（张佳宁 饰） 表白，突如其来的演唱会事故却让一切变成了一场无可挽回的悲剧，没想到的是，痛苦无助的张万森竟意外重启了这个夏天，再次回到悲剧发生前的林北星身边，而重启夏天的秘密，仿佛没有想象中那么简单……这一次，拼尽全力的张万森能否成功守护林北星，让所有刻骨铭心的遗憾都得以圆满？星河流转中的某个瞬间，青春里的那场绵绵大雪，能不能落在相爱的两人身上？', '内容源': 'yinhe,youku,tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 437\n\n{'ID': 'fntllf7o8800', '标题': '奇迹·笨小孩', '分类': '电影', '标签': '社会,生活', '看点': '成长,亲情,人生,正能量,贺岁档,励志,治愈', '演员': '易烊千玺,田雨,Helen哈琳', '导演': '文牧野', '发布年份': '2022.0', '适合年龄段': '19-34岁男', '情绪': '可缓解孤独,失望,沮丧,焦虑情绪。', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.795', '简介': '晴天,多云时适合观看。可缓解孤独,失望,沮丧,焦虑情绪。二十岁的景浩独自带着年幼的妹妹来到深圳生活，兄妹俩生活温馨却拮据。为了妹妹高昂的手术费，机缘巧合之下，景浩得到一个机会，本以为美好生活即将来临，却不料遭遇重创。在时间和金钱的双重压力下，毫无退路的景浩决定孤注一掷，而他陷入困境的平凡人生，又能否燃起希望的火花？电影《奇迹》是中宣部国家电影局2021年重点电影项目，也是2021年重点建党百年献礼片，描述十八大以后新时代年轻人在深圳创业的影片。', '内容源': 'tencent,youku,yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 231\n\n{'ID': 'l6mcp3208800', '标题': '燃冬', '分类': '电影', '标签': '爱情', '看点': '文艺,成长,人生,治愈', '演员': '周冬雨,刘昊然,屈楚萧', '导演': '陈哲艺', '发布年份': '2023.0', '适合年龄段': '19-34岁女', '情绪': '可缓解孤独,失望情绪。', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.911', '简介': '下雪时适合观看。可缓解孤独,失望情绪。延吉是中国北方边境的一个小镇，从上海来参加婚礼的浩丰感到有些迷茫。一次偶然的机会，他遇到了娜娜，一个让他着迷的年轻导游。她把他介绍给一个厨师朋友韩萧，他们三个人很快就因为韩萧的一顿饭而结下了友谊。这次热烈的接触使他们面对自己的过去和秘密，他们沉睡的欲望慢慢解开，就像长白山的风景和雪林一样。', '内容源': 'tencent,yinhe,youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 716\n\n{'ID': 'cnjflbvccc00', '标题': '猪猪侠大电影·恐龙日记', '分类': '电影', '标签': '动画,儿童,喜剧', '看点': '成长,萌,励志', '演员': '陆双', '导演': '钟彧', '发布年份': '2021.0', '适合年龄段': '4-6岁男', '情绪': '可缓解孤独,失望,沮丧情绪。', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.762', '简介': '晴天,多云时适合观看。可缓解孤独,失望,沮丧情绪。以阿五寻找猪猪侠为开端，讲述了见习超星特工阿五前往恐龙世界，结识了性格各异的伙伴们，在旅途中大家遭遇恶劣环境、恐龙袭击等种种危机，最后能够勇敢直面自己的弱点，相信伙伴的力量，从中得到了成长。', '内容源': 'tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3533\n\n{'ID': 'k9ndo48c8800', '标题': '极寒之城', '分类': '电影', '标签': '动作,悬疑', '看点': '硬汉', '演员': '夏雨,李立群,谭凯', '导演': '杨枫', '发布年份': '2023.0', '适合年龄段': '19-34岁男', '情绪': 'nan', '天气': '阴天,下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.787', '简介': '阴天,下雪时适合观看。凛冬子夜，枪手小顾追踪到蛰伏在裁缝铺里的杀手老焦，二人试探对峙，一段尘封多年的惨案，在二人的交锋中逐渐浮出水面。影片的故事背景锁定在民国三十四年（1945年）日军投降后的中国东北滨城。这段时间的滨城陷入无人管理的真空期，黑势力跋扈横行。奎叔、李桓、老苟、仲震、老贵等众多人物粉墨登场。在暗无天日，物欲横流的旧时代，一个关于坚守承诺，舍身成仁的史诗传奇故事也就此拉开帷幕。', '内容源': 'yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 2161\n\n{'ID': 'j7sqbf5c4400', '标题': '超凡动物奇观', '分类': '纪录片', '标签': '自然,科普,国外', '看点': '国家地理,探秘', '演员': '本尼迪克特·康伯巴奇', '导演': '马特·罗斯', '发布年份': '2022.0', '适合年龄段': '19-34岁男', '情绪': 'nan', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.824', '简介': '晴天,多云时适合观看。该纪录剧集由詹姆斯·卡梅隆监制，由荣获奥斯卡奖提名和英国电影学院奖的本尼迪克特·康伯巴奇配音。《超凡动物奇观》大胆创新，采用前沿电影制作技术，为您揭秘身怀秘技，感官超强的奇异动物。让我们从蜜蜂的视角感受花朵，偷听海象的对话，和发光松鼠一起深夜翱翔。', '内容源': 'youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1979\n\n{'ID': 'r8s52l40000', '标题': '天亮之前', '分类': '电影', '标签': '爱情,犯罪', '看点': '人生,逃亡', '演员': '郭富城,杨子姗,郝蕾', '导演': '吴中天', '发布年份': '2016.0', '适合年龄段': '19-34岁男', '情绪': '可缓解焦虑,失望情绪。', '天气': '阴天,雨天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.717', '简介': '阴天,雨天时适合观看。可缓解焦虑,失望情绪。赌徒高野继承千万资产后输得一干二净，家道也因此中落，就在他一筹莫展时，打扮妖冶的堕落天使茉茉主动上门提供特殊服务。不料他们却遭遇庄家买凶追杀，蒙眼飙车、女儿被绑一系列意外接踵而至。嗜赌成性的赌徒和用爱洗白的风情女，天亮之前他们该如何选择接下来的路', '内容源': 'tencent,youku,yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 2758\n\n{'ID': 'abdmb2t00400', '标题': '长津湖', '分类': '电影', '标签': '战争,历史,怀旧,军事', '看点': '主旋律,真实事件改编,正能量,抗美援朝,催泪', '演员': '吴京,易烊千玺,段奕宏', '导演': '陈凯歌', '发布年份': '2021.0', '适合年龄段': '35-54岁男', '情绪': 'nan', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.946', '简介': '下雪时适合观看。电影《长津湖》以抗美援朝战争第二次战役中的长津湖战役为背景，讲述了一段波澜壮阔的历史：71年前，中国人民志愿军赴朝作战，在极寒严酷环境下，东线作战部队凭着钢铁意志和英勇无畏的战斗精神一路追击，奋勇杀敌，扭转了战场态势，打出了军威国威。', '内容源': 'yinhe,tencent,youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3142\n\n{'ID': 'lviemo9s8800', '标题': '白昼冷光', '分类': '电影', '标签': '动作,惊悚', '看点': 'nan', '演员': '亨利·卡维尔 , 薇洛妮卡·恩切圭 , 布鲁斯·威利斯', '导演': '马布鲁克·埃尔·梅奇', '发布年份': '2012.0', '适合年龄段': '19-34岁男', '情绪': 'nan', '天气': '冰雹时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.2', '简介': '冰雹时适合观看。亨利·卡维尔 / 薇洛妮卡·在华尔街经营一家公司的年轻商人威尔（亨利·卡维尔 Henry Cavill 饰）从百忙之中抽身，飞往西班牙一处海滨小镇与家人度假。事隔多年，他与父亲马丁（布鲁斯·威利斯 Bruce Willis 饰）、母亲、弟弟乔什（莱菲·盖夫隆 Rafi Gavron 饰）及其女友达 拉（艾玛·汉密尔顿 Emma Hamilton 饰）相聚。家人围坐餐桌，其乐融融。但是祥和的气氛很快便被破坏，威尔接到公司破产的电话，心情糟糕到极点，由此也与家人的关系紧张起来。航海途中达拉意外受伤，当威尔去镇上买药期间，留在船上的家人却遭到绑架。焦急的威尔在当地无人可信，他独自搜索家人的下落，在此过程中竟发现了关于父亲和事件背后的诸多秘密……恩切圭 / 布鲁斯·威利斯', '内容源': 'tencent,yinhe,youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1489\n\n{'ID': 'n97rndb88800', '标题': '银河写手', '分类': '电影', '标签': '喜剧,搞笑,生活,院线,社会', '看点': '人生,成长', '演员': '宋木子,合文俊,李飞', '导演': '李阔', '发布年份': '2024.0', '适合年龄段': 'nan', '情绪': '可缓解孤独,失望,沮丧情绪。', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.736', '简介': '晴天,多云时适合观看。可缓解孤独,失望,沮丧情绪。两个心怀梦想的小编剧，张了一和孙谈，写出了自认为可以名垂青史的原创电影剧本《七秒人》，在一群狐朋狗友的帮助下，寻找资方和制片人，试图将电影搬上大银幕，开启了一段啼笑皆非的“寻梦”之旅。这是一出编剧生存图鉴，也是当代打工人生存启示录。', '内容源': 'tencent,yinhe,youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 2805\n\n{'ID': 'm7768uoc4400', '标题': '冰雪奇缘2', '分类': '动漫', '标签': '冒险,奇幻,动画,歌舞', '看点': 'nan', '演员': '克里斯汀·贝尔,伊迪娜·门泽尔,乔什·盖德', '导演': '珍妮弗·李', '发布年份': '2019.0', '适合年龄段': 'nan', '情绪': 'nan', '天气': '雾天,下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.543', '简介': '雾天,下雪时适合观看。历经严酷考验，阿伦戴尔王国终于回归往日平静。艾莎女王、安娜公主以及他们的好友雪宝、克里斯托弗、驯鹿斯文过着平静安逸的生活。可是最近一段时间，艾莎总会被一段神秘的吟唱所困扰，为了追寻真相，她义无反顾踏上了征途。担心姐姐的安全，安娜和雪宝、克里斯托弗他们紧紧跟随。在那座常年被浓雾所笼罩的森林里，不仅藏着神秘的自然力量，更隐藏着关于阿伦戴尔王国、艾莎的魔法来源以及两位公主父母丧生等一系列的秘密。', '内容源': 'youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1695\n\n{'ID': 'm569d824cc00', '标题': '无间毒票', '分类': '电影', '标签': '动作,犯罪,警匪', '看点': '缉毒,警察,卧底,史诗,热血', '演员': '陈龙,吕良伟,曾志伟', '导演': '王筱刚', '发布年份': '2024.0', '适合年龄段': 'nan', '情绪': '可缓解愤怒,焦虑情绪。', '天气': '阴天,雨天时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.778', '简介': '阴天,雨天时适合观看。可缓解愤怒,焦虑情绪。电影通过描写一个智勇双全的警察贤俊，靠着冷静果敢，成功取得了具有化学天赋的毒枭——华哥的信任，卧底进入其犯罪集团，全程取证掌握了他们研究制作贩运买卖新型毒品——“邮票”的全过程，最终靠着他的机智和协助警方的配合，成功破获了华哥为首的贩毒制毒团伙。', '内容源': 'tencent,youku,yinhe'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 3224\n\n{'ID': 'g99n54q0cc00', '标题': '少年歌行', '分类': '电视剧', '标签': '武侠,古装', '看点': '优酷独播剧', '演员': '李宏毅,刘学义,林博洋', '导演': '尹涛', '发布年份': '2022.0', '适合年龄段': '19-34岁男', '情绪': 'nan', '天气': '下雪时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.829', '简介': '下雪时适合观看。身着千金裘却连客栈维修都没钱的抠门老板萧瑟与初入江湖的雷门弟子雷无桀相识于雪落山庄，之后二人共闯江湖。途中阴错阳差与天外天少主无心、雪月城枪仙之女司空千落、唐门大弟子唐莲和大将军之女叶若依结识，却也意外卷入江湖、庙堂纷争，众人逐渐发现这其中之事与这位名为萧瑟的客栈老板有着千丝万缕的关系，期间他们夺黄金棺材、破生死困局、寻海外仙山、战天下第一、破千军万马，萧瑟的身份也逐渐明朗，原来这一切都与12年前的皇位之争有关。最终在少年们共同努力之下，击碎了天启皇城中的那些阴谋诡谲，也揭开了迟到了12年的真相……', '内容源': 'youku'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 415\n\n{'ID': 'bv4mqsds0400', '标题': '七号房的礼物', '分类': '电影', '标签': 'nan', '看点': '亲情,成长,豆瓣高分,温馨', '演员': 'Aras Bulut Iynemli', '导演': '梅米特·艾达·厄兹泰金', '发布年份': '2019.0', '适合年龄段': '55岁以上女', '情绪': '可缓解悲伤,孤独,失望,愤怒,焦虑,震惊,怀疑,羞愧,自责,悔恨,尴尬情绪。', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.679', '简介': '晴天,多云时适合观看。可缓解悲伤,孤独,失望,愤怒,焦虑,震惊,怀疑,羞愧,自责,悔恨,尴尬情绪。一个被错误指控谋杀的精神病父亲和他可爱的六岁女儿之间的温情故事。改编自2013年韩国电影《7号房的礼物》。', '内容源': 'yinhe,tencent'}"], ["有没有适合下雪天观看的电影", "file_path: /var/folders/2d/05gc4k5n1jv4rwftqk5p4hq40000gp/T/tmpx7b1bp9e/34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_name: 34ea6f595b258a233afe4add30551158_df_multi_col_data_media1.csv\nfile_type: text/csv\nfile_size: 2119527\ncreation_date: 2024-07-05\nlast_modified_date: 2024-07-05\nrow_number: 1674\n\n{'ID': '83cnhvs00400', '标题': '冰糖炖雪梨', '分类': '电视剧', '标签': '爱情,青春,真人秀,竞技', '看点': '甜宠,体育,小说改编,成长,优酷独播剧,运动', '演员': '吴倩,张新成,周历杰', '导演': '朱锐斌', '发布年份': '2020.0', '适合年龄段': '19-34岁女', '情绪': '可缓解孤独,失望,沮丧情绪。', '天气': '晴天,多云时适合观看。', '行业': 'nan', '声纹': 'nan', 'rating_score': '0.65', '简介': '晴天,多云时适合观看。可缓解孤独,失望,沮丧情绪。小学里，胆小怯懦的男孩黎语冰一直被“大王同桌”棠雪欺负，他们唯一的共同之处，是都有一个滑冰的梦想。数年后，当他们在“霖大”校园重逢时，怯懦的黎语冰，已经逆袭成了霖大冰球队的“冰神”，而“大王同桌”棠雪，却是个被调剂到兽医专业，早已远离梦想的大学生。重逢后的“冰神”黎语冰腹黑上线，让儿时的“施虐”大王变成了自己的“受虐”小助理。在这场胶着的报复行动中，黎语冰渐渐发现，棠雪坚韧的性格，保护朋友的“大王”气质，都让棠雪这个人闪闪发光，而自己打冰球最倚赖的秘诀勇气，正是来自棠雪，而棠雪也在黎语冰的启发下，找回了曾经失去的速滑梦想。两人终于互通心意，但随着小时候的误会爆发，花滑天才喻言介入，棠雪初恋边澄回归，与棠雪不合的校花周染干扰，父母阻挠，比赛受伤，运动生涯面临抉择，一个个压力接踵而来。在爱与梦想交织的冰面上，两个热血盎然的心将爱化为动力，滑向未知而悸动的人生。', '内容源': 'youku'}"]]
            samples = self.tokenizer(
                sentence_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            ).to("cuda")
            #).to("cpu")
        else:
            samples = get_calib_dataset(
                data=self.calib_data,
                tokenizer=self.tokenizer,
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                split=self.split,
                text_column=self.text_column,
            )
            samples = torch.cat(samples, dim=0)

        inps = []
        attention_mask = []
        layer_kwargs = {}

        best_device = get_best_device()
        #modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    attn_mask = args[1]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                attention_mask.append(attn_mask)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            if isinstance(self.model, XLMRobertaForSequenceClassification):
                ret = self.model(**samples, return_dict=True)
                print(ret)
            else:
                self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        if not isinstance(self.model, XLMRobertaForSequenceClassification):
            layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        if "input_ids" in layer_kwargs:
            layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]
        attention_mask = attention_mask[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        return modules, layer_kwargs, inps, attention_mask

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        if self.awq_model.model_type == "deepseek_v2":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        device = next(layer.parameters()).device
        self.inps = self.inps.to(device)  # in case multi-gpu
        self.attention_mask = self.attention_mask.to(device)    
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        # !!! now self.inps only contain input_ids, 
        # should also contain attention_mask
        #import pdb; pdb.set_trace()
        self.inps = self._module_forward(self.inps, layer, module_kwargs, self.attention_mask)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs
