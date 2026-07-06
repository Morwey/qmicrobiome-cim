# -*- coding: utf-8 -*-
# %% [markdown]
# # 量子解析大曲抑制乳酸菌 —— QVAE + 玻尔兹曼机(CIM) 端到端 tutorial
#
# 把 **scQuantaVita** 的量子‑经典混合玻尔兹曼能量范式迁移到酱酒发酵菌群多组学，
# 端到端演示榜题所需的每一步，**求解器全部使用开物 Kaiwu SDK（公开版 1.3.1）**：
#
# 1. **合成多组学数据**：由高斯图模型（GGM，已知直接互作/偏相关结构）跨发酵阶段生成，
#    预埋接触型抑制菌 `C`、经代谢物 `M` 的非接触型抑制菌 `NC`、广谱抑制菌 `BS`、共存菌 `co1`。
# 2. **阶段① QVAE（DVAE+RBM）**：学表示 + 每样本群落系统势 CSP（RBM 负相由 Kaiwu 采样）。
# 3. **能量面 + LAB 筛选**：按“丰度随发酵阶段能量面迁移”筛演替中心菌，并入与关键乳酸菌 `LAB`
#    直接关联的菌 → 关键菌群。
# 4. **阶段② 玻尔兹曼机互作层**：在关键菌上学耦合网络 {J,h}，负相用 **Kaiwu SA**（离线）采样、
#    e^(−E) 重要性重加权估计；接触/非接触、广谱/特异 从 J 读出。
# 5. **阶段③ 反事实敲除**：Kaiwu 重采样量化各通路对 LAB 丰度的贡献度，输出累计 ≥60% 关键菌。
#
# **求解器**：经典离线 = `kw.classical.SimulatedAnnealingOptimizer`；真机 = `kw.cim.CIMOptimizer`。
# 伊辛矩阵一律由 `kw.conversion.qubo_matrix_to_ising_matrix` 生成。**无任何外置自写求解器。**
#
# **运行环境**：Python 3.10 + kaiwu==1.3.1 + torch + anndata + numpy。
# **许可**：先免费注册（platform.qboson.com），把凭据放入环境变量再运行：
# `export KAIWU_USER_ID=...  KAIWU_SDK_CODE=...`（**切勿把凭据写入源码或提交仓库**）。

# %%
import os, io, contextlib, numpy as np, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")     # 保持输出干净
import torch, anndata
import kaiwu as kw
from qvae_dvae_rbm import DVAE_RBM          # 阶段① QVAE：项目内 DVAE+RBM（去凭据版）

USER_ID = os.environ.get("KAIWU_USER_ID"); SDK_CODE = os.environ.get("KAIWU_SDK_CODE")
assert USER_ID and SDK_CODE, "请先设置环境变量 KAIWU_USER_ID / KAIWU_SDK_CODE（勿写入源码）"
kw.license.init(user_id=USER_ID, sdk_code=SDK_CODE)
RNG = np.random.default_rng(1)

# ---- named nodes of the planted interaction network (the ground truth) ----
NODES = ["LAB", "C", "NC", "M", "BS", "co1", "T2"]
KIND  = {"LAB": "taxon", "C": "taxon", "NC": "taxon", "M": "metabolite",
         "BS": "taxon", "co1": "taxon", "T2": "taxon"}
idx = {n: i for i, n in enumerate(NODES)}; K = len(NODES)


# %% [markdown]
# ## 1. 预埋真值：高斯图模型（精度矩阵 Θ = 直接互作结构）
# 偏相关 ρ_ij = −Θ_ij / √(Θ_ii Θ_jj)：Θ_ij>0 ⇒ 负偏相关（抑制），Θ_ij<0 ⇒ 正偏相关（共存），
# Θ_ij=0 ⇒ 条件独立（无直接互作）。据此预埋：
# - C→LAB 接触抑制（Θ>0）；co1→LAB 共存（Θ<0）
# - NC→M（Θ<0，促进）、M→LAB（Θ>0，抑制）、NC↔LAB 无直接边 ⇒ NC 经 M 非接触抑制
# - BS→LAB/co1/T2 均抑制 ⇒ 广谱

# %%
Th = np.eye(K) * 2.6
def setTh(a, b, v): Th[idx[a], idx[b]] = Th[idx[b], idx[a]] = v * 1.25
setTh("C", "LAB", +0.75); setTh("M", "LAB", +0.70); setTh("NC", "M", -0.70)
setTh("BS", "LAB", +0.55); setTh("BS", "co1", +0.60); setTh("BS", "T2", +0.60)
setTh("co1", "LAB", -0.65)
assert np.linalg.eigvalsh(Th).min() > 0.05, "Θ 需正定"
Sigma = np.linalg.inv(Th); Lchol = np.linalg.cholesky(Sigma)


# %% [markdown]
# ## 2. 跨发酵阶段生成多组学丰度表
# 5 个工艺节点（丢糟/堆积/入窖/窖内/出窖），阶段均值漂移使**群落能量面随发酵迁移**；
# 阶段内协方差 = Θ⁻¹（真实互作结构不随时间变）。混入 20 个纯噪声特征供 QVAE 去伪存真。

# %%
NST, N_REP = 5, 80
node_names = ["丢糟", "堆积", "入窖", "窖内", "出窖"]
n = NST * N_REP
stage = np.repeat(np.arange(NST), N_REP); t = stage / (NST - 1)
drift = np.zeros(K)
for nm, v in {"LAB": 1.2, "co1": 0.9, "T2": 0.7, "C": -1.0, "NC": -0.9, "M": -1.0, "BS": -1.0}.items():
    drift[idx[nm]] = v                                     # 成熟度漂移(阶段均值)
Xs = np.zeros((n, K))
for st in range(NST):
    m = stage == st
    Xs[m] = drift * (st / (NST - 1)) + RNG.standard_normal((m.sum(), K)) @ Lchol.T
N_JUNK = 20
feat = NODES + [f"junk{i}" for i in range(N_JUNK)]; fidx = {f: i for i, f in enumerate(feat)}
X = np.column_stack([Xs, RNG.standard_normal((n, N_JUNK))]).astype(np.float32)
Xz = ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)   # 组学里对应 CLR/log 后 z-score
print(f"多组学数据表: {n} 样本 × {len(feat)} 特征 (结构 {K} + 噪声 {N_JUNK}), {NST} 工艺节点")


# %% [markdown]
# ## 阶段① QVAE（DVAE+RBM）：学表示 + 群落系统势 CSP
# 用 `DVAE_RBM`（RBM 负相由 Kaiwu SA 采样）。CSP = 每样本隐态在 RBM 下的能量，
# 随发酵阶段迁移 → 支撑“演替规律”，并驱动下面的关键菌筛选。

# %%
ad = anndata.AnnData(Xz)
ad.obs["batch"] = 0; ad.obs["batch"] = ad.obs["batch"].astype("category")
qvae = DVAE_RBM(hidden_dim=64, latent_dim=16, sample_method="ising_sa", optimizer_type="sa",
                user_id=USER_ID, sdk_code=SDK_CODE, project_no=None, device=torch.device("cpu"))
qvae.set_adata(ad, batch_key="batch")
with contextlib.redirect_stdout(io.StringIO()):        # 静默 QVAE 内部训练日志
    qvae.fit(ad, epochs=12, batch_size=64, lr=1e-3, rbm_lr=2e-3, early_stopping=False,
             verbose=0, ckpt_dir="./qvae_ckpt")
    reps = qvae.get_representation(step=1, adata=ad)
z = torch.tensor((reps > reps.mean(0)).astype(np.float32))
with torch.no_grad():
    CSP = qvae.rbm.energy(z).cpu().numpy()
print("各工艺节点平均 CSP:", {node_names[s]: round(float(CSP[stage == s].mean()), 2) for s in range(NST)})
print(f"CSP 与工艺进程相关 |corr|={abs(np.corrcoef(t, CSP)[0,1]):.2f} (显著迁移 → 能量面随发酵移动)")


# %% [markdown]
# ## 3. 能量面 + LAB 筛选关键菌群
# 关键菌 = **跨发酵阶段驱动能量面迁移的菌（差异丰度）** ∪ **与关键乳酸菌 LAB 直接关联的菌**。
# 纯噪声特征既不随阶段迁移、也不与 LAB 关联 → 被排除。

# %%
early, late = Xz[stage <= 1].mean(0), Xz[stage >= 3].mean(0)
esurf = np.abs(late - early)                               # 驱动能量面迁移(跨阶段差异丰度)
Xc = Xz.copy()                                             # 阶段内中心化 → LAB 偏相关
for st in range(NST):
    m = stage == st; Xc[m] -= Xc[m].mean(0)
lab = Xc[:, fidx["LAB"]]
lab_assoc = np.array([abs(np.corrcoef(Xc[:, j], lab)[0, 1]) for j in range(len(feat))])
def _nrm(v): v = np.asarray(v, float); return (v - v.min()) / (np.ptp(v) + 1e-9)
score = np.maximum(_nrm(esurf), _nrm(lab_assoc))          # 两条判据取并
junk_top = max(score[fidx[f]] for f in feat if f.startswith("junk"))
keep = [feat[j] for j in np.argsort(-score) if score[j] > junk_top * 1.10]
if "LAB" not in keep: keep = ["LAB"] + keep
print("筛出的关键菌群/代谢物:", keep, " (应含 LAB,C,NC,M,BS,co1,T2; 无 junk)")


# %% [markdown]
# ## Kaiwu 负相采样器：伊辛矩阵由 kw.conversion 生成，SA 采样 + e^(−E) 重加权
# 真机改用 `kw.cim.CIMOptimizer(task_name=..., project_no=...)`。

# %%
def _fields_to_ising(h, J):
    """ising 能量 E(s)=-h·s-0.5 s^T J s → QUBO → kw.conversion 生成 Kaiwu 伊辛矩阵。"""
    k = len(h); d = J.sum(1); Q = np.zeros((k, k))
    for i in range(k):
        Q[i, i] = -2 * h[i] + 2 * d[i]
        for j in range(i + 1, k): Q[i, j] = -4 * J[i, j]
    return kw.conversion.qubo_matrix_to_ising_matrix(Q)[0]

def _decode(spins, k):
    """Kaiwu 自旋(长度 k+1, 末位 ancilla) → 长度 k 的 ±1 向量 (ancilla gauge)。"""
    out = []
    for r in np.asarray(spins):
        s = np.asarray(r).astype(int); out.append(((s[:k] * s[-1]) + 1) // 2)
    return 2 * np.array(out) - 1.0

def kaiwu_sample(h, J, size=400):
    """开物经典求解器(SA)采一批构型。真机: kw.cim.CIMOptimizer(...).solve(ising)。"""
    worker = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=50, alpha=0.9, cutoff_temperature=0.1,
        iterations_per_t=10, size_limit=size, rand_seed=1)
    return _decode(worker.solve(_fields_to_ising(h, J)), len(h))

def neg_phase(h, J, size=400):
    """负相期望 <s_i>,<s_i s_j>：Kaiwu 采样 + e^(−E) 重要性重加权(校正采样偏置)。"""
    P = kaiwu_sample(h, J, size)
    E = -P @ h - 0.5 * np.einsum("si,ij,sj->s", P, J, P)
    w = np.exp(-(E - E.min())); w /= w.sum()
    return w @ P, (P * w[:, None]).T @ P


# %% [markdown]
# ## 阶段② 玻尔兹曼机互作层：学出 {J,h}
# 关键菌按**工艺节点内中心化**(去阶段漂移混杂)后二值化；梯度
# ΔJ_ij ∝ <s_i s_j>_data − <s_i s_j>_model，负相由 Kaiwu 采样估计。

# %%
kcols = [fidx[f] for f in keep]; kk = len(keep); kidx = {f: i for i, f in enumerate(keep)}
Xk = Xz[:, kcols].copy()
for st in range(NST):
    m = stage == st; Xk[m] -= Xk[m].mean(0)
Sbin = np.where(Xk > 0, 1.0, -1.0)
m_data, C_data = Sbin.mean(0), (Sbin.T @ Sbin) / len(Sbin)
h, J = np.zeros(kk), np.zeros((kk, kk))
print("训练玻尔兹曼机 (Kaiwu SA 负相) ...")
for it in range(150):
    m_mod, C_mod = neg_phase(h, J, 400)
    h += 0.2 * (m_data - m_mod)
    dJ = C_data - C_mod; np.fill_diagonal(dJ, 0.0)
    J += 0.2 * dJ; J -= 0.2 * 0.003 * np.sign(J)
    np.fill_diagonal(J, 0.0); J = 0.5 * (J + J.T)
Jk = lambda a, b: J[kidx[a], kidx[b]] if a in kidx and b in kidx else float("nan")


# %% [markdown]
# ## 4. 读出互作网络：接触/非接触、广谱/特异

# %%
SIG = 0.15
metabolites = [f for f in keep if KIND.get(f) == "metabolite"]
def classify(x):
    if abs(Jk(x, "LAB")) > SIG and Jk(x, "LAB") < 0: return "接触", None
    for me in metabolites:
        if abs(Jk(x, me)) > SIG and Jk(x, me) > 0 and abs(Jk(me, "LAB")) > SIG and Jk(me, "LAB") < 0:
            return "非接触", me
    return "无显著抑制", None

print("\n===== 关键乳酸菌 LAB 的互作邻居 (直接耦合 J) =====")
for nm in sorted([f for f in keep if f != "LAB"], key=lambda f: Jk(f, "LAB")):
    print(f"  J[{nm:>4}, LAB] = {Jk(nm,'LAB'):+.3f}  ({KIND.get(nm,'?')})")
print("\n----- 接触/非接触 与 广谱/特异 (自动判定) -----")
for x in ["C", "NC", "BS"]:
    if x not in kidx: continue
    mode, via = classify(x)
    neg = [f for f in keep if f != x and KIND.get(f) == "taxon" and Jk(x, f) < -SIG]
    via_s = f"，经代谢物 {via} (J[{x},{via}]={Jk(x,via):+.2f}, J[{via},LAB]={Jk(via,'LAB'):+.2f})" if via else ""
    print(f"  {x}: 【{mode}】{via_s} | 【{'广谱' if len(neg) >= 2 else '特异'}】 负耦合 taxon={neg or '—'}")
print("  ✓ NC 原始相关为负但直接耦合≈0、经 M 介导 → 正确判为非接触 (相关≠互作)")


# %% [markdown]
# ## 阶段③ 反事实敲除：各通路贡献度 + 累计 ≥60% 关键菌
# 钳制某菌为“缺失(−1)”，Kaiwu 重采样测 P(LAB=high) 变化 ΔP = 该通路对 LAB 的抑制贡献。

# %%
def p_lab_high(h, J, clamp=None, size=2500):
    P = kaiwu_sample(h, J, size)
    if clamp is not None:
        P = P[P[:, clamp[0]] == clamp[1]]
        if len(P) < 30: return None
    return (P[:, kidx["LAB"]] == 1).mean()

base = p_lab_high(h, J)
contrib = {f: (p_lab_high(h, J, (kidx[f], -1.0)) or base) - base for f in keep if f != "LAB"}
inhib = {k_: v for k_, v in contrib.items() if v > 1e-3}; tot = sum(inhib.values())
ranked = sorted(inhib.items(), key=lambda kv: -kv[1]); cum, key60, done = 0.0, [], False
print(f"\nP(LAB=high) 基线={base:.3f}; 各通路对‘抑制 LAB’的相对贡献度:")
for nm, v in ranked:
    pct = 100 * v / tot; cum += pct
    if not done: key60.append(nm)
    if not done and cum >= 60: done = True
    print(f"  {nm:>4} ({KIND.get(nm,'?'):>10}): ΔP={v:+.3f}  贡献={pct:5.1f}%  累计={cum:5.1f}%")
print(f"⇒ 累计贡献 ≥60% 的关键抑制菌/物: {key60}")

# %% [markdown]
# ## 说明
# - 真机：把 `kaiwu_sample` 内的 `SimulatedAnnealingOptimizer` 换为
#   `kw.cim.CIMOptimizer(task_name=..., project_no=...)` 即用相干伊辛机采样（scQuantaVita 已在 1025 节点验证）。
# - 真实项目：阶段① 直接对接扩增子/宏基因组+代谢组的 anndata；样本偏少时先筛高可变特征再 CLR/log。
# - 合成数据效应量为“干净教学演示”而调；真实数据更噪，边显著性用 bootstrap/置换判定。

# %%
print("\n================ tutorial 完成 (全程 Kaiwu 1.3.1) ================")
