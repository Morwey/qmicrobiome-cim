# 酱香微生物分析-量子方案（QVAE/BM）

对接"科技攻关·揭榜挂帅"第六期 **04 号榜题**：*酱香型白酒高温大曲抑制乳酸菌的接触与非接触效应解析研究*。
用玻色量子已在单细胞工作 **scQuantaVita** 中验证的"量子‑经典混合玻尔兹曼能量模型"范式，
把菌群多组学的"谁抑制谁、接触还是非接触、贡献多少"变成一套可计算、可读出、可湿实验验证的模型量。

## 文件
| 文件 | 内容 |
|---|---|
| `报告_量子解析大曲抑制乳酸菌.md` | **系统报告**：量子出发点、方法框架、与验收指标逐条映射、里程碑与风险 |
| `qmicrobiome_tutorial.py` | **可运行 tutorial**：端到端 QVAE→玻尔兹曼机→贡献度 |
| `qmicrobiome_tutorial.ipynb` | 同上的 **Jupyter/Colab 版**（已带执行输出，Colab「全部运行」即可） |

## 快速开始
```bash
pip install numpy
pip install kaiwu-1.3.1-*.whl   # 开物公开版 1.3.1 wheel（platform.qboson.com 免费注册获取）
```
切换到开物Kaiwu量子套件：
```bash
# 代码内：BACKEND = "kaiwu"，并先 kw.license.init(user_id=..., sdk_code=...)
```
转 Jupyter/Colab：`jupytext --to notebook qmicrobiome_tutorial.py` 或直接在 Colab 按 `# %%` 分块粘贴。

## Tutorial 会证明什么（合成数据预埋真值，可复现）
1. **QVAE 抽关键菌群**：从含 20 个噪声特征的 29 维表里，置换检验干净地选出 7 个结构特征（LAB/C/NC/M/BS/co1/T2），排除噪声。
2. **玻尔兹曼机学互作网络** {J,h}，负相期望由 **开物 CIM 采样**（离线可用 numpy 精确枚举对表验证一致）。
3. **接触 vs 非接触**（榜题核心）：
   - `C`：直接负耦合 `J[C,LAB]≈-0.68` ⇒ **接触型**
   - `NC`：原始 `corr(NC,LAB)≈-0.68` 为负，但直接 `J[NC,LAB]≈-0.08`≈0，且 `J[NC,M]>0, J[M,LAB]<0` ⇒ **非接触/代谢物 M 介导**
   - ✓ 玻尔兹曼机把间接相关扣除——**相关≠互作**，正确分离接触/非接触
4. **广谱 vs 特异**：`C`=特异（仅 LAB）、`BS`=广谱（LAB/co1/T2 多靶）、`NC`=特异。
5. **贡献度 + ≥60% 关键菌**：反事实敲除（CIM 重采样）量化各通路对 LAB 丰度的相对贡献，输出累计 ≥60% 的关键抑制菌集 `{C, BS, M}`（≈80%）。
6. **群落系统势 CSP**：每样本能量刻画群落热力学稳定度（低=稳定相干态，高=竞争受挫态）。

## 与验收指标的对应（详见报告）
| 榜题指标 | 交付物 | Tutorial 对应步骤 |
|---|---|---|
| ≥10 定植微生物、≥100 代谢物、≥4 节点×3 轮 | QVAE 关键特征抽取 + CSP 演替 | ① |
| ≥1 关键乳酸菌、≥10 互作微生物、广谱/特异、接触/非接触、长/短 | BM 学 `J` + 结构判据 | ②④ |
| 各通路相对贡献度、验证方法、后验 ≥60% 关键菌 | 反事实敲除 + 湿实验闭环 | ③ |
| ≥2 SCI + ≥1 专利 | CSP‑community 新指标 + 贡献度量化法（专利） | — |

## 依赖
- 必需：`numpy`
- 可选：`kaiwu`（开物 SDK 公开版 1.3.1，真·CIM）、`torch`（附录中忠实 QVAE，可选）
- 环境参考：Python 3.10，`numpy==2.2.6`（开物 1.3.1 会自动 pin 该版本）

## 关键参考
- scQuantaVita 手稿（QBoson × 广州国家实验室）：CIM 玻尔兹曼采样 + 经典 NN 训练，1025 节点 ≥10⁵× 加速。
- 开物 Kaiwu SDK（公开版 1.3.1）：`kw.conversion.qubo_matrix_to_ising_matrix`、`kw.cim.SimulatedCIMOptimizer`(离线 CIM 模拟)、`kw.cim.CIMOptimizer`(真·相干伊辛机)。
- 后端设计同构参考：QDock‑Kaiwu 的 `kw_backend.py`（kaiwu/reference 双后端）。
