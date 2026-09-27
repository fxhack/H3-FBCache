# -*- coding: utf-8 -*-
"""h3_fb_cache.py — MiniMax H3 视频 DiT 的 First-Block-Cache 加速补丁。

把 WaveSpeed / FBCache 的"首块缓存"机制做成 H3 专用实现：
每步采样时，若 DiT block 1 的输入与上一步足够相似（相对 L1 距离 < 阈值），
则直接复用上一步末块（最后一个 DiTBlock）的输出、跳过 block 1..N-1 的全部
计算——命中的步只算 block 0，速度接近跳步而无需减少步数。

安装
----
把本文件放进 ComfyUI 的 custom_nodes 目录，重启 ComfyUI。
启动 ComfyUI 前设置环境变量开启（不设 = 关闭，不影响任何人）：

    Windows :  set H3_FBCACHE=0.25
    Linux   :  export H3_FBCACHE=0.25

阈值含义：相对 L1 距离阈值，越大跳得越狠、越快，质量风险越高。
0.2~0.3 是常用区间（0.25 为推荐默认）。换 prompt / 换分辨率导致
序列形状变化时自动 miss，安全。

实测（Ryzen AI Max+ 395 / Radeon 8060S, gfx1151, ROCm 7.2）：
    480p 864x480  6 步：采样 179s -> 131s
    768p 1360x768 6 步：采样 827s -> 540s（总时长 -28%）
命中步约 1.5~2.5s（正常步 29~130s）。逐帧目视验收无伪影。

注意
----
* 缓存命中会改变采样轨迹：同 seed 出片的构图与关闭时不同（机制使然）。
* 要完全确定性输出：set H3_FBCACHE=0（或不设该环境变量）。
* 只作用于 MiniMax H3 的 DiT blocks，不影响其它模型（LTX/Flux/SD 等）。
* 纯 PyTorch 实现，与显卡无关（NVIDIA / AMD / Apple 均可）。

依赖：ComfyUI 需带 MiniMax H3 支持（comfy/ldm/minimax/model.py 存在，
ComfyUI v0.3.7x+）。没有该支持时本补丁静默跳过，不影响 ComfyUI 运行。
"""
import os

_THRESH = float(os.environ.get("H3_FBCACHE", "0") or 0)

if _THRESH <= 0:
    print("[h3_fb_cache] 未启用（set H3_FBCACHE=0.25 开启，0.25 为推荐阈值）", flush=True)
else:
    try:
        import torch
        import comfy.ldm.minimax.model as _m
    except Exception as _e:
        print("[h3_fb_cache] 跳过：本机 ComfyUI 无 MiniMax H3 支持或环境异常（%r）" % _e,
              flush=True)
        _m = None

if _THRESH > 0 and _m is not None:
    # 跨步持久缓存（形状不匹配时自动 miss；进程内多模型各自覆盖，逻辑安全）
    _STATE = {"b1_in": None, "final_out": None}

    def _rel_l1(a, b):
        return ((a.float() - b.float()).abs().mean() / (b.float().abs().mean() + 1e-6)).item()

    _orig_block_forward = _m.DiTBlock.forward

    def _fbc_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        opts = transformer_options if transformer_options is not None else {}
        idx = opts.get("block_index", -1)
        st = opts.get("_h3fbc")
        if st is None:
            st = {"hit": False}
            opts["_h3fbc"] = st

        if idx == 0:
            st["hit"] = False  # 每次前向从 block 0 重置单次调用状态
            return _orig_block_forward(self, x, t_emb, mod_segments, rope_freqs,
                                       transformer_options=opts, attention=attention)

        if idx == 1:
            cached = _STATE.get("b1_in")
            if cached is not None and cached.shape == x.shape:
                try:
                    rel = _rel_l1(x, cached)
                except Exception:
                    rel = 1.0
                if rel < _THRESH:
                    fo = _STATE.get("final_out")
                    if fo is not None and fo.shape == x.shape:
                        st["hit"] = True
                        return fo.to(dtype=x.dtype)
            _STATE["b1_in"] = x.detach().clone()
            return _orig_block_forward(self, x, t_emb, mod_segments, rope_freqs,
                                       transformer_options=opts, attention=attention)

        if st.get("hit"):
            return x  # 命中：透传（block 1 已返回上一步的末块输出）
        out = _orig_block_forward(self, x, t_emb, mod_segments, rope_freqs,
                                  transformer_options=opts, attention=attention)
        n = getattr(self, "_h3_n_blocks", None)
        if n is not None and idx == n - 1:
            _STATE["final_out"] = out.detach().clone()
        return out

    _orig_model_init = _m.MiniMaxH3Model.__init__

    def _model_init(self, *args, **kwargs):
        _orig_model_init(self, *args, **kwargs)
        n = len(self.blocks)
        for b in self.blocks:
            b._h3_n_blocks = n

    _m.MiniMaxH3Model.__init__ = _model_init
    _m.DiTBlock.forward = _fbc_forward
    print("[h3_fb_cache] 已生效：相对L1阈值=%s（block1 输入相似则复用上一步末块输出）" % _THRESH,
          flush=True)

# ComfyUI 要求自定义节点声明这两个符号，否则整文件会被标为 IMPORT FAILED。
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
