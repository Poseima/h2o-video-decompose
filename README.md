# H2O Video Decompose

> 参考视频 → 逐帧拆解 → 可编辑复刻

给 Codex 等 AI Agent 使用的视频拆解 Skill。输入一段视频，可以查到镜头何时切换、动效如何分层、音效和音乐卡在哪一帧。需要复刻时，可把测量结果用于 Blender、After Effects 或 Remotion，并用实际导出的成片逐帧核对。

## 快速开始

### 1. 安装 Skill

有仓库访问权限后，在你的项目目录运行以下命令并选择使用的 Agent。

```bash
npx skills add Poseima/h2o-video-decompose --skill h2o-video-decompose-distribution
```

私有仓库需要 GitHub 凭据。也可以克隆仓库，将其中的 `h2o-video-decompose-distribution` 目录复制到 Agent 的自定义 Skill 目录。

### 2. 安装运行依赖

基础拆解使用 Python 3.12 和 FFmpeg。音频分轨与自动拍点、3D Gaussian Splatting（3DGS）有各自的可选依赖。macOS（Apple Silicon）已验证这三条链路，Windows 11（x64）已验证基础拆解和二维短片复刻，音频模型与 3DGS 待实测。当前支持范围为 macOS 和 Windows。详见[安装与验证](h2o-video-decompose-distribution/INSTALL.md)。

### 3. 把视频交给 Agent

```text
使用 $h2o-video-decompose-distribution 拆解我提供的视频。
输出全片镜头时间线、音效与配乐卡点，以及关键段试听。
```

也可以指定片段和任务，例如逐帧测量 `00:02–00:05` 的推镜、打字和拖影，再用 Remotion 重建这一段。只想先生成故事板和帧索引的话，安装基础依赖后可在 Skill 目录运行：

```bash
python scripts/decompose.py scan "input.mp4" "work/scan"
```

## 能做什么

| 工作流 | 可以拿到什么 |
| --- | --- |
| 全片拆解 | 故事板、候选切点、镜头时间线和源帧索引。可变帧率视频按实际帧时间戳核对。 |
| 2D / 2.5D 动效精测 | 运镜和图层轨迹、入场与回弹时序、局部清晰度变化，以及按需测量的字形、排版与色板。 |
| 3D 场景拆解 | 相机投影与位姿估计、可见几何、逐物件材质、灯光与合成线索。可为 Blender 建模、布光和渲染验收提供参数。 |
| 声画拆解 | 对齐的原混音、可用的对白／音乐／音效分轨、声画对应表、节拍候选和关键段试听。 |
| 可编辑复刻与验收 | 先评估指定范围的可复刻性，再按镜头和图层重建。用成片检查运镜、接缝、光色突变、停帧和声画同步，并记录仍可见的差异。 |
| 写实静态场景 3DGS | 可选的 COLMAP → Brush 重建路线，输出可导入三维工具的 PLY。可用新视角受原片相机平移幅度与场景稳定性限制。 |

复刻以指定范围的 1:1 还原为验收目标，需要宿主支持子代理协作。UI 动效、排版和硬表面场景通常更适合参数化重建，真人表演、复杂角色和重度物理特效的高精度复刻超出当前能力。详见[能力边界](h2o-video-decompose-distribution/references/capabilities.md)。

## 案例

- **白页打字与连续推拉** · 从快切和页面更新中拆出打字、误选与局部清晰度变化。[猫学长前十秒案例](h2o-video-decompose-distribution/references/cat-opening-reconstruction.md)
- **港风街景和霓虹灯牌** · 测量招牌几何、浮雕分层、体积雾和辉光，用于 Blender 场景复刻。[港风三维案例](h2o-video-decompose-distribution/references/hk-neon-3d-reconstruction.md)
- **石牌坊静态场景** · 查看 Apple Silicon 上的 3DGS 重建实测，以及原片运镜如何限制新视角。[3DGS 基准](h2o-video-decompose-distribution/references/pavilion-3dgs-benchmark.md)

## 文档

- [Skill 入口](h2o-video-decompose-distribution/SKILL.md)
- [使用手册与命令参数](h2o-video-decompose-distribution/references/workflow.md)
- [镜头连续性验收](h2o-video-decompose-distribution/references/continuity.md)
- [局部质感诊断](h2o-video-decompose-distribution/references/local-appearance.md)
