# 安装与验证

把整个 `h2o-video-decompose-distribution` 目录放入宿主的自定义 Skill 目录，通过「分发版」或 `$h2o-video-decompose-distribution` 调用。以下命令从 Skill 根目录运行。也可以在项目目录中使用脚本的完整路径。示例中的 `work` 是测试输出目录，实际项目请替换为自己的路径。

基础拆解适用于扫查、镜头精测、连续性验收、局部质感对比和 `scene3d` 三维测量。需要对白、音乐、音效分轨或自动拍点时，加装音频。需要训练可换视角的高斯泼溅模型时，加装 3DGS。

同一个分发包适用于 macOS 和 Windows。Linux 暂不支持。

| 平台 | 实测范围 |
| --- | --- |
| macOS（Apple Silicon） | 已验证基础拆解、音频分轨与拍点、COLMAP 重建和 Brush PLY 导出。 |
| Windows 11（x64） | 已验证基础扫查、中文与空格路径、逐帧精测与标注、二维短片复刻及验收报告生成。可选音频模型和 3DGS 待实测。 |

## 基础拆解

### 安装

安装 [Python 3.12](https://www.python.org/downloads/) 和带 `libx264` 的 FFmpeg。终端应能直接运行 `ffmpeg` 和 `ffprobe`。

| 平台 | FFmpeg 安装方式 |
| --- | --- |
| macOS | 有 Homebrew 时运行 `brew install ffmpeg`。也可选用官方链接的预编译包。 |
| Windows | 从 FFmpeg 下载页选择 Windows 构建，解压后把包含 `ffmpeg.exe` 和 `ffprobe.exe` 的目录加入 `PATH`，再打开终端。 |

[FFmpeg 官方下载](https://ffmpeg.org/download.html)

macOS

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements.txt
```

Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r scripts/requirements.txt
```

如果 PowerShell 禁止执行激活脚本，直接用 `.\.venv\Scripts\python.exe` 替换安装与验证命令中的 `python`。基础依赖采用无界面版 OpenCV，请在独立虚拟环境安装。如果本机额外配置的索引有连接问题，可使用 `python -m pip --isolated install --index-url https://pypi.org/simple -r scripts/requirements.txt`，或在可联网的环境下载匹配目标系统与 Python 版本的 wheel，再用 `python -m pip install --no-index --find-links wheels -r scripts/requirements.txt` 离线安装。

### 验证

生成一段带合成音轨的测试视频，验证 FFmpeg 编码、FFprobe 读帧以及扫查。复测时使用新的输出目录。

```text
python --version
python -c "import numpy, cv2, scipy, PIL, matplotlib; print('Python dependencies OK')"
ffmpeg -version
ffprobe -version
python -c "from pathlib import Path; Path('work').mkdir(exist_ok=True)"
ffmpeg -nostdin -f lavfi -i "testsrc2=size=320x180:rate=24" -f lavfi -i "sine=frequency=440:sample_rate=48000" -t 8 -c:v libx264 -pix_fmt yuv420p -c:a aac -n "work/smoke.mp4"
python scripts/decompose.py scan "work/smoke.mp4" "work/scan"
```

确认 `work/scan` 中有 `source.json`、`frames.csv` 和故事板图。正式使用时，把测试片换成用户提供的视频。分发版的新扫查记录用 SHA-256 校验原片。跨目录或机器续测时，传入同一文件的新路径即可。由旧版本生成的扫查在原路径仍可使用，换目录后需重新扫查。

## 可选音频

### 安装

先完成基础安装，再安装 Git，并确认 `git --version` 可以运行。继续使用同一个 Python 3.12 虚拟环境。

```sh
# Apple Silicon macOS
python -m pip install torch==2.8.0 torchaudio==2.8.0
```

```sh
# Windows CPU
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

安装两个包的同版本构建。其他系统或架构先确认官方有对应的 Python 3.12 wheel。Bandit 自动使用可用的 Apple MPS，其他环境可用 CPU。 [PyTorch 2.8 安装说明](https://pytorch.org/get-started/previous-versions/#v280)

```text
python -m pip install -r scripts/requirements-audio.txt
python -c "from pathlib import Path; Path('models').mkdir(exist_ok=True)"
```

下载 Bandit 的 `checkpoint-multi.ckpt` 到 `models/`，或放到自行选择的目录。运行时用 `--checkpoint` 传实际路径，也可设 `H2O_BANDIT_CHECKPOINT`。脚本加载前会核对权重 SHA-256。Beat This! 默认首次联网下载 `final0`，可以设 `TORCH_HOME` 指定缓存位置。离线时用 `--beat-checkpoint` 或 `H2O_BEAT_CHECKPOINT` 指向已下载的本地权重。 [Bandit 模型下载](https://zenodo.org/records/12701995) · [Beat This!](https://github.com/CPJKU/beat_this)

### 验证

```text
python -c "import torch, torchaudio, soundfile; from eluate.core.separator import BanditSeparator; from beat_this.inference import File2Beats; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
python scripts/soundtrack.py prepare "work/smoke.mp4" "work/audio" --scan "work/scan/source.json" --checkpoint "models/checkpoint-multi.ckpt" --device cpu
```

打开 `work/audio/soundtrack.json`。分轨成功时 `status` 为 `ready`，混音、对白、音乐、音效对应的 WAV 均可读。只有混音或部分轨道时，查看 `notes` 中的模型、权重或解码错误。合成单音适合验证分轨链路。验证拍点时，另用一段节奏清晰的实际视频重新扫查和运行 `prepare`，检查 `beat_status` 和 `beats.csv`。对拍点时间再做试听复核。

## 可选 3DGS

### 安装

1. 保留基础环境。装 COLMAP 4.2 或兼容版本，确认 `colmap` 可以从 `PATH` 调用。macOS 可用 `brew install colmap`。Windows 使用官方构建并把原生 `colmap.exe` 所在目录加入 `PATH`。其他安装方式见[COLMAP 官方手册](https://colmap.github.io/install.html)。
2. 下载与系统和架构匹配的 Brush CLI，解压到自选目录。macOS 如无执行权限，对解压的 `brush_app` 执行 `chmod +x`。脚本先用 `--brush`，其次用 `H2O_BRUSH_BIN`，最后从 `PATH` 查找 `brush_app` 或 `brush`。项目基准使用 v0.3.0。 [Brush v0.3.0 官方发行包](https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0)
3. 只在需要编辑或渲染时安装 Blender 和与其版本匹配的 KIRI 3DGS Render 扩展。用 Blender 的首选项从磁盘安装扩展 ZIP，启用后导入 `splat/scene_final.ply`。渲染前先用一帧试渲染确认扩展与 GPU 可用。Apple Silicon / Metal 渲染步骤见[使用手册 3DGS 小节](references/workflow.md#4-高精写实静态场景-3dgs-本地重建与-blender-52-渲染gs3d_pipelinepy)。 [Blender 下载](https://www.blender.org/download/) · [KIRI 扩展发行包](https://github.com/Kiri-Innovation/3dgs-render-blender-addon/releases)

### 验证

先检查命令行。以下假定 Brush 已加入 `PATH`。若使用独立下载目录，用二进制的实际路径执行 `--help`。PowerShell 执行带引号的程序路径时，在路径前加 `&`。

```text
colmap -h
colmap feature_extractor -h
brush_app --help
python scripts/gs3d_pipeline.py --help
python scripts/gs3d_pipeline.py --video "work/smoke.mp4" --out-dir "work/3dgs-frames" --start 0 --end 1 --fps 8 --stage frames
```

确认 `work/3dgs-frames/frames_clean` 有抽出的图片，`gs3d_report.json` 的 `frames.kept_count` 大于零。合成测试图只用于抽帧验证。完整验证使用一段约 5 秒、相机实际平移、纹理丰富且光照稳定的静态场景视频。下例的 `tools/brush_app` 是示例位置。Windows 使用实际 `.exe` 路径，Brush 已在 `PATH` 时可省略末尾参数。

```text
python scripts/gs3d_pipeline.py --video "work/static-scene.mp4" --out-dir "work/3dgs-run" --start 0 --end 5 --fps 12 --max-side 960 --steps 2500 --max-splats 100000 --stage all --brush "tools/brush_app"
```

检查 `gs3d_report.json` 的 `colmap.registered_images` 和 `brush.final_ply`，确认 `splat/scene_final.ply` 存在并能打开。若分阶段排查，先对真实片段运行 `--stage frames`，再使用同一输出目录依次运行 `--stage colmap` 和 `--stage brush`。Brush 训练需要可用 GPU 和相应驱动。Blender 的导入与渲染独立于该脚本执行。
