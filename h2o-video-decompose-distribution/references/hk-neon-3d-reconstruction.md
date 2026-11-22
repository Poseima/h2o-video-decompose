# 《ICE-Hong Kong（feat. 季候风）》25 秒港风三维街景与霓虹灯牌逆向复刻

这份案例使用 `decompose.py scene3d` 对第 738 帧（**24.60 秒**，1080 × 1080，30 fps）做场景范式判别与六维三维逆向精测，指导 Blender 5.2 Cycles 复刻。

## 一、 场景范式自动判别与全画幅三维诊断

~~~bash
python scripts/decompose.py scene3d "项目/ICE-Hong Kong（feat. 季候风）.mp4" \
  "项目/输出/hk_scene3d" --frame 738 --window 8 \
  --corners "569,362;1112,471;1112,661;569,533" --rect-size "860,280" --symmetric-complete
~~~

1. **范式判别（`scene_paradigm_classification`）**：
   - `detected_paradigm = "3D_BLENDER_SCENE"`（`score_3d_scene = 6.8` vs `score_2d_planar = 0.0`）。
   - 判据：连续曲面与体积散射明暗梯度、三维街道平行线消失点汇聚、多深度层光流速度视差（前景悬挑灯牌、中景双层巴士与车流、后景唐楼立面）以及强光源径向指数衰减光晕。
2. **相机与关键三维构件位姿（`pillar_1_camera_and_projection`）**：
   - 投影模式 `PERSP`，36mm 等效焦距 `35.0 mm`；
   - 前景「茶樓酒家」灯牌左缘高度 `171.0 px`、右缘高度 `190.0 px`，实测 **左右透视缩短深度比 $D_L / D_R = 1.1111$**（在中心基准深度 $9.0\text{ m}$ 下，左缘深度 `9.45 m`、右缘深度 `8.50 m`）；
   - 相机空间四角 3D 顶点坐标：
     - `P_TL = [0.2608, 1.6052, -9.4500]`
     - `P_TR = [4.6247, 0.5586, -8.5000]`
     - `P_BR = [4.6247, -0.9794, -8.5000]`
     - `P_BL = [0.2608, 0.0672, -9.4500]`
   - 显式景深对焦距离 `explicit_focus_distance_m = 8.95 m`。

## 二、 几何拓扑、分层高度遮罩与扩边背板（`pillar_2_geometry_and_topology`）

1. **体素/方块阵列周期**：
   - 正交展平纹理检出像素阶跃特征（`texture_sampling_mode = "Closest"`，原生体素步长 `4 px`），水平方向检出 11 格方块接缝周期（`step_x_px ≈ 78 px`）。
2. **三层前推式 3D 网格构建**：
   - **底层灯箱（`z_offset = 0.0m`）**：深琥珀褐底板（`Roughness = 0.45`），背面加 `-0.12m` 铁皮底盒与右侧 3 根生锈角铁悬挑支架；
   - **中层装饰灯管（`z_offset = +0.008m`，`Displace = +0.024m`）**：由 `mask_mg_tubes.png` 驱动，包含 11 格分段红黄双层外框、上下双排金色波浪祥云与顶部中央翠绿空心五角星；
   - **前层书法字灯管（`z_offset = +0.022m`，`Displace = +0.036m`）**：由多尺度高斯差分脊线（DoG Ridge）提取的 `mask_fg_tubes.png` 驱动，完整保留「茶 樓 酒 家」双线空心灯管内腔，并物理悬浮在黄色祥云灯管前方。
3. **不透明基底物理遮挡与扩边干净背板**：
   - 前层与中层灯管均采用深色 `Principled BSDF`（`Base Color = (0.02, 0.01, 0.005)`）配合脊线 `Alpha` 遮罩阻断后层光线，防止红字与黄云交叠处发生纯加法叠加过曝；
   - 局部单构件复刻使用的后景参考背板 `bg_clean.png` 按实测光晕衰减长度自动外扩 `+20 px`，移动前景 3D 灯牌展示完整「家」字与右边框时绝无残留边框重影。

## 三、 各向异性体积雾与合成器黑柔镜管线（`pillar_4` ~ `pillar_6`）

| 渲染与光学环节 | 实测指标 | Blender 5.2 节点配置 |
| --- | --- | --- |
| **色彩管理与单层发光** | 灯芯到边缘饱和度回落比 `0.78`，符合 `AgX` 曲线 | `view_transform = 'AgX'`（`AgX - Medium High Contrast`）；底板 `Emission = 1.04`、中层灯管 `1.06`、前层红字 `1.08`，配合 `Gamma = 1.06~1.08` |
| **潮湿水汽体积雾** | 光晕指数衰减长度 $\lambda_{\text{fog}} \approx 9.5\text{ px}$，近场光雾强而远景暗部深邃（`p05 = 0.012`） | 全局体积立方体挂载 `Principled Volume`：`Density = 0.0052`、**`Anisotropy = 0.78`**、暖琥珀色 `(0.76, 0.64, 0.48)`，配 `38W` 红色散射面光 |
| **合成器黑柔镜与色散** | 四角红蓝径向亚像素偏移 $\Delta r_{R-B} \approx 1.1\text{ px}$；暗部暖褐 `Lift #281B0F`、高光朱红/琥珀 `Gain #F25B2A` | `scene.compositing_node_group` 串联：`Glare (Fog Glow, Threshold=0.62, Strength=0.14)` $\to$ `Blur(15px) + Mix(SCREEN, Factor=0.05)` $\to$ `Lens Distortion (Distortion=0.003, Dispersion=0.005, Fit=True)` |
