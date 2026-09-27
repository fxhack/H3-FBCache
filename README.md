# H3-FBCache — MiniMax H3 视频生成首块缓存加速补丁

给 ComfyUI 的 **MiniMax H3**（Hailuo H3 开源视频模型）加一个"首块缓存"加速层，
移植自 [WaveSpeed / FBCache](https://github.com/chengzeyi/Comfy-WaveSpeed) 的机制，
针对 H3 的模型结构（音视频打包双流 + fused 内核 + prefetch 队列）做了专用适配。

纯 PyTorch 实现，**与显卡无关**（NVIDIA / AMD / Apple 均可用），不依赖任何
CUDA 专属组件（SageAttention、Triton、torch.compile 都不需要）。

---

## 原理

扩散模型多步采样时，相邻两步的 DiT 内部特征往往高度相似。本补丁在每步前向时
比较 **block 1 的输入** 与上一步的缓存：相对 L1 距离低于阈值，就认为后续所有
block 的贡献与上一步几乎相同——直接复用上一步末块输出、跳过 block 1..N-1 的
全部计算。命中的步只算 block 0，耗时约为正常步的 **2%**。

这是速度与质量的权衡开关：阈值越大越快、质量风险越高。

## 安装

1. 把 `h3_fb_cache.py` 放进 ComfyUI 的 `custom_nodes\` 目录
2. 启动 ComfyUI 前设置环境变量（不设 = 关闭）：

```cmd
:: Windows (cmd)
set H3_FBCACHE=0.25
python_embeded\python.exe main.py ...
```

```bash
# Linux / MacOS
export H3_FBCACHE=0.25
python main.py ...
```

3. 启动日志出现 `[h3_fb_cache] 已生效：相对L1阈值=0.25` 即安装成功。
   （没有该行 = 未开启；提示"无 MiniMax H3 支持" = ComfyUI 版本太旧。）

工作流 **不需要任何改动**：补丁作用于服务端的 H3 模型，所有 H3 工作流
（任意分辨率、任意 LoRA 组合）自动生效；其它模型（LTX / Flux / SD）不受影响。

## 阈值怎么调

| H3_FBCACHE | 行为 |
|---|---|
| 不设 / 0 | 关闭（确定性输出） |
| 0.2 | 保守：命中略少，最稳 |
| **0.25（推荐）** | 速度/质量平衡点 |
| 0.3 | 激进：略快，步间命中率波动大 |
| >0.35 | 不建议，质量劣化风险明显 |

实测（Ryzen AI Max+ 395 / Radeon 8060S, gfx1151, ROCm 7.2，5 秒 124 帧视频）：

| 档位 | 关闭 | 开启（0.25） | 变化 |
|---|---|---|---|
| 480p 864×480 6 步 | 采样 179s（热启动 ~240s） | 采样 ~131s（热 ~190s） | 采样 -27% |
| 768p 1360×768 6 步 | 采样 827s（热 945.7s） | 采样 540s（热 683.6s） | **总时长 -28%** |

分辨率的规律：分辨率越高（attention 占比越大）收益越大。命中步耗时仅
1.5~2.5s（正常步 29~130s），6 步 schedule 下典型命中 2~4 步。

## 注意事项（必读）

1. **轨迹会变**：缓存命中改变采样轨迹，同 seed 出片的构图与关闭时不同。
   这是机制本身，不是 bug。要复现确定性画面：`set H3_FBCACHE=0`。
2. **重要片子建议关闭**或用 0.2 低阈值；批量试 prompt 时开着省时间。
3. 换 prompt / 换分辨率导致序列形状变化时自动 miss，不会出错。
4. 与 int8 attention 等其它优化**可叠加**（本补丁只跳 block，不改算子）。

## 已知兼容性

* 需要 ComfyUI 自带 MiniMax H3 支持（`comfy/ldm/minimax/model.py` 存在，
  ComfyUI v0.3.7x 及以上）。旧版 ComfyUI 会打印"跳过"并正常启动。
* AMD ROCm / NVIDIA CUDA / Apple MPS 均可（无硬件专属内核）。
* 与 ComfyUI 的 `--fast`、`--highvram`、`--cache-lru` 等参数无冲突。

## 文件

| 文件 | 说明 |
|---|---|
| `h3_fb_cache.py` | 补丁本体，放进 `ComfyUI/custom_nodes/` 即可 |
| `README.md` | 本说明 |

## 许可

自由使用与转载。如果对你有用，也欢迎把改进反馈回来。
