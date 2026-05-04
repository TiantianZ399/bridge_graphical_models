# 一个更本质的统一：把 Diffusion、Rectified Flow、Brownian Bridge 和 Poisson Flow 看成 Path-space VAE

> 核心观点：Diffusion、Rectified Flow、Flow Matching、Brownian/Schrödinger Bridge、Poisson Flow/PFGM 不应该被看成几种彼此竞争的采样技巧。它们更像是同一个连续时间潜变量图模型的不同选择：选择什么 coupling、什么 bridge、什么几何、什么 stochastic gauge，以及如何把一个 non-causal path 投影成 causal generative dynamics。

我更愿意把这个家族称为：

$$
\boxed{\textbf{Path-space VAE with probability-current gauge}}
$$

或者更工程一点：

$$
\boxed{\textbf{Graphical Bridge Generative Model}}
$$

这篇文章想解释为什么我认为这个视角比“Diffusion vs Rectified Flow”更接近本质。

---

## 1. 现有的两个统一方向：Brownian bridge 和 Poisson flow

现在文献里确实已经有两条重要路线在做统一。

第一条是 Brownian bridge / Schrödinger bridge / stochastic interpolants 路线。它把生成问题写成“找一条从简单分布到数据分布的随机过程”。如果参考过程是 Brownian motion，那么 Schrödinger bridge 就是在所有满足端点分布约束的 path measure 中，找一个离 Brownian reference 最近的过程。近几年的 Diffusion Schrödinger Bridge Matching、simulation-free Schrödinger bridge、Diffusion Flow Matching、Unified Bridge Algorithm 都在推进这条线。[12][13][14][16][17]

第二条是 Poisson flow / electrostatic field 路线。PFGM 把数据点看成增广空间中 $z=0$ 超平面上的电荷，生成一个由 Poisson equation 决定的 electric field，再沿 field line 从简单分布流回数据分布。[10] PFGM++ 进一步把数据嵌入 $N+D$ 维空间：$D=1$ 时退化为 PFGM，$D\to\infty$ 时趋向 diffusion model。[11] 2025–2026 年的 Field Matching 和 Flow/Field Matching duality 又进一步把 electrostatic field 与 conditional flow matching 连接起来。[18][19]

但我觉得这两条路还没有触及最本质的问题。Brownian bridge 统一的是 reference path geometry；Poisson flow 统一的是 augmented-space field geometry。它们都很漂亮，但还不是最一般的“生成图模型”语言。

真正该问的是：

> 一个生成模型到底在学习什么 latent object？是一个 $z$？是一串 $x_1,\dots,x_T$？还是整条 path $X_{0:1}$？

从这个角度看，diffusion 本来就应该被看成一种 VAE：它是一个有很多层 latent variables 的 hierarchical latent-variable model，并且 DDPM 本来就是用 variational bound 训练的。[1][2] 连续时间版本还可以被写成 infinitely deep VAE / path-space variational inference。[3]

---

## 2. 最一般的对象：不是 ODE，也不是 SDE，而是一条 latent path

设数据分布为

$$
\pi_1 = \pi_{\mathrm{data}},
$$

简单 prior/base distribution 为

$$
\pi_0,
$$

例如标准 Gaussian。

传统生成方向是：

$$
Z_0\sim \pi_0, \qquad X_1\sim \pi_{\mathrm{data}}.
$$

但是中间不是空的。中间应该有一整条 latent path：

$$
X_{0:1}=\{X_t:0\le t\le 1\}.
$$

这条 path 才是 diffusion、rectified flow、Brownian bridge、Poisson flow 共同拥有的潜变量结构。

最干净的图模型可以写成：

$$
Z_0 \longrightarrow X_{0<t<1} \longrightarrow X_1.
$$

或者用 VAE 的语言写成：

- encoder / inference path：给定 data point $x$，构造一条从 prior side 到 $x$ 的 path；
- decoder / generative path：从 $z\sim \pi_0$ 出发，沿一个 causal dynamics 生成 $x$。

这比“到底用 ODE 还是 SDE”更基本。

---

## 3. Endpoint coupling：先决定谁和谁配对

第一步不是选 neural network，而是选 endpoint coupling：

$$
\Gamma(dz,dx)\in \Pi(\pi_0,\pi_1),
$$

其中 $z\sim\pi_0$，$x\sim\pi_1$。

不同方法在这里隐含了不同选择：

- Diffusion 常常等价于一种 independent/noisy coupling：data 被 Gaussian noise 腐蚀。
- Rectified Flow 常从独立 coupling 或 OT/minibatch OT coupling 出发，再学习一条尽量直的流。[5][6][8]
- Schrödinger bridge 使用 entropic OT / path-space KL 决定 coupling 与 path measure。[12][13][14]
- VAE 则显式学习一个 encoder coupling $q_\phi(z|x)\pi_{\mathrm{data}}(dx)$。

我的观点是：**coupling 是生成模型里被严重低估的对象**。如果 coupling 错了，后面再漂亮的 ODE/SDE 也只是在拟合错误的语义配对。

---

## 4. Bridge kernel：再决定中间路径怎么走

给定 endpoint $(z,x)$，选择一个 conditional path distribution：

$$
R_\lambda^{z,x}(dX_{0:1}),\qquad X_0=z,\quad X_1=x.
$$

这就是 bridge kernel。

然后把所有 endpoint pair 混合起来，得到 teacher path measure：

$$
Q_{\Gamma,\lambda}(dX_{0:1})
=
\int R_\lambda^{z,x}(dX_{0:1})\,\Gamma(dz,dx).
$$

这个 $Q_{\Gamma,\lambda}$ 是整个理论的中心。不同模型只是选了不同的 $R$：

### Rectified Flow

最简单的 deterministic bridge：

$$
X_t=(1-t)z+tx,
$$

于是

$$
\dot X_t=x-z.
$$

Rectified Flow 的训练就是让 neural ODE 的 velocity 尽量匹配这个 conditional velocity。[5][6]

### Diffusion / Gaussian bridge

传统 diffusion 常写成 data-to-noise 方向：

$$
X_t=\alpha_t X_0+\sigma_t\varepsilon,
\qquad \varepsilon\sim \mathcal N(0,I).
$$

这是一类 Gaussian bridge。预测 noise、预测 score、预测 denoised data、预测 velocity，本质上是同一个 conditional expectation 的不同参数化。[4][7]

### Brownian / Schrödinger bridge

选择 Brownian motion 或 Brownian bridge 作为 reference，然后在端点分布约束下做 path-space KL minimization。这个选择把 diffusion model 和 entropic optimal transport 放在一起。[12][13]

### Poisson / electrostatic bridge

把状态空间增广为

$$
\mathcal M = \mathbb R^N\times \mathbb R^D,
$$

把数据分布放在额外维度的零平面上，令其成为“电荷分布”，再用 Poisson equation 诱导一个 vector field。PFGM++ 的 $D$ 正是一个从 PFGM 到 diffusion 的几何连续参数。[10][11]

所以更本质的问题不是“要 diffusion 还是 rectified flow”，而是：

$$
\boxed{\text{What bridge kernel should define the latent path?}}
$$

---

## 5. Path-space VAE：把 diffusion 和 flow 都写成 ELBO

现在我们定义一个 generative decoder path measure：

$$
P_\theta(dY_{0:1}),
$$

例如由一个 SDE 或 ODE 给出：

$$
dY_t=b_\theta(Y_t,t)\,dt+\sigma_\eta(Y_t,t)\,dB_t,
\qquad Y_0\sim \pi_0.
$$

最一般的训练目标可以写成 path-space KL：

$$
\min_\theta\; 
\mathrm{KL}\left(Q_{\Gamma,\lambda}(X_{0:1})\;\|\;P_\theta(Y_{0:1})\right).
$$

如果要写成更标准的 VAE，可以引入 encoder：

$$
q_{\phi,\lambda}(z,X_{0:1}|x)
= q_\phi(z|x)R_\lambda^{z,x}(dX_{0:1}),
$$

以及 decoder：

$$
p_\theta(x,z,X_{0:1})
=p_0(z)p_\theta(X_{0:1}|z)p_\theta(x|X_1).
$$

于是 ELBO 是

$$
\log p_\theta(x)
\ge
\mathbb E_{q_{\phi,\lambda}}
\left[
\log p_\theta(x,z,X_{0:1})
-
\log q_{\phi,\lambda}(z,X_{0:1}|x)
\right].
$$

Diffusion 是一种固定 encoder、超多 latent layers 的 VAE；Rectified Flow 是 zero-temperature / deterministic path limit；Schrödinger bridge 是带 Brownian reference 的 path-space VAE；PFGM 是增广空间里的 field-induced path VAE。

这不是说所有模型已经在文献里被完全写成同一个 ELBO，而是说：**这是一个统一它们的自然数学语言**。

有意思的是，2025–2026 年已经出现非常接近这个方向的论文：Latent Stochastic Interpolants 直接把 stochastic interpolants 放进 learned encoder、decoder 和 latent-space ELBO 中，明确提出用单一 continuous-time ELBO 联合学习 latent representation 与 generative process。[20] 这和“把 diffusion/flow 当作 VAE family”非常接近。

---

## 6. 为什么 Flow Matching / Rectified Flow 会自然出现？

假设 teacher path $Q$ 下的局部速度是 $u_t$。如果 path 是 deterministic interpolation，那么

$$
u_t = \dot X_t.
$$

如果 path 是 stochastic process，那么 $u_t$ 可以理解为它的 drift。

我们要把这个 non-causal teacher path 变成一个 causal Markov dynamics。最优 Markov velocity 是条件期望：

$$
v^\star(x,t)=\mathbb E_Q[u_t\mid X_t=x].
$$

因此自然得到 regression loss：

$$
\mathcal L(\theta)
=
\int_0^1
\mathbb E_Q
\left[
\|u_t-v_\theta(X_t,t)\|^2
\right]dt.
$$

这就是 Flow Matching / Rectified Flow 的核心。Flow Matching 用 fixed conditional probability paths 做 vector field regression；Rectified Flow 选择直线 interpolation，所以 target 是 $x-z$。[5][7]

如果 $Q$ 是非 Markov 的，这一步叫 Markovian projection：用一个 Markov process 去匹配原过程的一时刻 marginals。2024 年的 Diffusion Flow Matching 理论论文正是从 coupling + deterministic/stochastic bridge 出发，把它们定义为 path measure，再学习其 Markovian projection drift。[16]

所以 Rectified Flow 的本质不是“直线”本身，而是：

$$
\boxed{
\text{non-causal bridge}\quad \Rightarrow \quad \text{causal Markov projection}
}
$$

直线只是一个特别简单、特别高效的 bridge choice。

---

## 7. ODE 和 SDE 不是本质差别，而是 probability-current gauge

设 $\rho_t$ 是中间 marginal density。任何生成过程首先要满足 continuity equation：

$$
\partial_t\rho_t+\nabla\cdot J_t=0,
$$

其中 $J_t$ 是 probability current。

如果写成 ODE：

$$
dX_t=v_t(X_t)dt,
$$

那么

$$
J_t=\rho_t v_t.
$$

如果写成 SDE：

$$
dX_t=b_t(X_t)dt+\sigma_t(X_t)dB_t,
\qquad a_t=\sigma_t\sigma_t^\top,
$$

Fokker--Planck equation 给出

$$
J_t=ho_t b_t-rac12\nabla\cdot(a_tho_t).
$$

因此，给定同一个 probability current $J_t=\rho_t v_t$，可以构造无穷多个 SDE drift：

$$
b_t^{(a)}(x)
=
v_t(x)+
\frac{1}{2\rho_t(x)}\nabla\cdot(a_t(x)\rho_t(x)).
$$

特别地，如果

$$
a_t(x)=g(t)^2I,
$$

则

$$
b_t^{(a)}(x)=v_t(x)+\frac{g(t)^2}{2}\nabla\log\rho_t(x).
$$

这条公式说明：

- $a_t=0$：ODE / flow / rectified flow；
- $a_t>0$：SDE / diffusion；
- score $\nabla\log\rho_t$：是从 ODE gauge 切换到 SDE gauge 时出现的 correction term。

Score-SDE 论文已经明确给出了 reverse-time SDE 和等价 probability flow ODE：二者可以产生同样的 marginal distribution path，只是一个随机、一个确定。[4]

所以 diffusion 和 rectified flow 的差别不是“一个有噪声，一个没噪声”这么简单。更准确地说：

$$
\boxed{
\text{它们是同一个 probability path/current 的不同 gauge 表达。}
}
$$

---

## 8. Gaussian bridge 下，score、noise、denoiser、velocity 是同一件事

在 diffusion 习惯方向下，设

$$
X_t=\alpha_tX_0+\sigma_t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
$$

令 $\rho_t$ 是 $X_t$ 的 density，score 为

$$
s_t(x)=\nabla_x\log\rho_t(x).
$$

由 Tweedie-type identity 可得：

$$
\mathbb E[\varepsilon\mid X_t=x]
=-\sigma_t s_t(x).
$$

又因为

$$
\dot X_t=\dot\alpha_tX_0+\dot\sigma_t\varepsilon,
$$

所以 conditional velocity 为

$$
v^\star(x,t)
=
\mathbb E[\dot X_t\mid X_t=x]
=
\frac{\dot\alpha_t}{\alpha_t}x
+
\left(
\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2
-
\dot\sigma_t\sigma_t
\right)s_t(x).
$$

这说明，在 Gaussian path 下，预测 score、预测 noise、预测 denoised data、预测 velocity，本质上是线性重参数化。Flow Matching 论文也指出 Gaussian probability paths 包含了已有 diffusion paths，并且可以用同样的 vector-field regression 训练 CNF。[7]

这也是为什么很多现代 diffusion / flow model 在工程上可以互相转换参数化：$\epsilon$-prediction、$x_0$-prediction、score prediction、$v$-prediction 背后都是同一个 conditional expectation。

---

## 9. Brownian bridge 和 Poisson flow 在统一框架里的位置

### Brownian bridge 是 entropic path geometry

Brownian bridge 路线可以写成：

$$
R^{z,x}=\text{Brownian bridge from }z\text{ to }x.
$$

混合所有 endpoints 得到：

$$
Q(dX_{0:1})=
\int R^{z,x}(dX_{0:1})\Gamma(dz,dx).
$$

如果进一步优化 $\Gamma$ 或优化整个 path measure，使其在满足端点 marginals 的同时最接近 Brownian reference，就得到 Schrödinger bridge。

这条路线的优点是数学干净：KL、Girsanov、entropic OT、Markovian projection 都可以放进来。缺点是它默认 Brownian reference 是合适的几何；但在图像、蛋白、语言、金融时间序列等任务里，真实 latent geometry 未必是 Brownian。

### Poisson flow 是 harmonic / electrostatic field geometry

PFGM 路线则先把数据变成电荷：

$$
\Delta \Phi = -\rho_{\mathrm{data}}\delta_{z=0},
\qquad
E=-\nabla\Phi.
$$

生成过程沿着 electric field 运动。PFGM++ 把额外维度从 $D=1$ 推广到任意 $D$，并发现 $D\to\infty$ 与 diffusion 对齐。[11]

在我的图模型语言里，PFGM/PFGM++ 不是 diffusion 的“竞争者”，而是选择了另一个 state space 和另一个 bridge geometry：

$$
\mathcal M=\mathbb R^N\times\mathbb R^D,
\qquad
\text{bridge由 harmonic/electrostatic field 诱导。}
$$

2026 年的 Flow/Field Matching duality 正在把这件事说得更清楚：CFM 从 data-space conditional probability path 出发；IFM/EFM 从 augmented-space interaction field 出发。它们在 forward-only IFM 子类上可以建立 bijection，但一般 IFM 更 expressive。[19]

这说明 Poisson/electrostatic 路线不是旁支，而是“field-based graphical bridge model”的一个非常重要候选。

---

## 10. 我认为的 guiding rules

如果要真正统一 diffusion 和 rectified flow，我会用下面几条规则。

### Rule 1：先定义 latent graphical model，再谈 sampler

先写：

$$
Z_0\to X_{0<t<1}\to X_1.
$$

不要一开始就争论 ODE、SDE、score、noise schedule。那些都是后面的 parameterization。

### Rule 2：显式选择 endpoint coupling

$$
\Gamma(dz,dx)\in\Pi(\pi_0,\pi_1).
$$

Independent coupling、OT coupling、entropic coupling、learned encoder coupling 都是不一样的 inductive bias。

### Rule 3：显式选择 bridge kernel

$$
R_\lambda^{z,x}(dX_{0:1}).
$$

直线、Gaussian bridge、Brownian bridge、Schrödinger bridge、Poisson field、manifold geodesic，都只是不同 bridge。

### Rule 4：把 non-causal bridge 投影成 causal Markov dynamics

$$
v^\star(x,t)=\mathbb E[u_t\mid X_t=x].
$$

这一步自然产生 Flow Matching / Rectified Flow loss。

### Rule 5：ODE/SDE 是 gauge，不是模型本体

$$
b^{(a)}=v+\frac{1}{2\rho}\nabla\cdot(a\rho).
$$

$a=0$ 是 deterministic flow；$a>0$ 是 diffusion；score 是 gauge correction。

### Rule 6：训练目标最好能回到 ELBO / path KL

$$
\min_\theta
\mathrm{KL}(Q_{\mathrm{encoder\ path}}\|P_{\theta,\mathrm{decoder\ path}}).
$$

这会把 diffusion 作为 VAE、rectified flow 作为 deterministic VAE、Schrödinger bridge 作为 Brownian-reference path VAE、PFGM 作为 augmented-field path VAE 统一起来。

---

## 11. 目前最接近这个方向的 active papers

截至 2026-05-01，我看到最相关的活跃线索大概是这些。

### A. Stochastic Interpolants：统一 flow 和 diffusion 的主干

Stochastic Interpolants 明确提出一个统一 flow-based 和 diffusion-based 方法的连续时间框架：用 stochastic process 在有限时间内 bridge 任意两个 densities，并且同一 density path 同时满足 transport equation 和带可调 diffusion coefficient 的 forward/backward Fokker--Planck equation。[9]

它已经非常接近本文的 probability-current gauge 观点。差别是：它更多从 probability path/PDE 角度组织，而不是从 VAE graphical model 角度组织。

### B. Diffusion Flow Matching / Markovian projection

2024 年的 Theoretical Guarantees in KL for Diffusion Flow Matching 把 FM / stochastic interpolants / rectified flows 视为：固定 coupling + deterministic/stochastic bridge 定义 path measure，再学习其 Markovian projection drift。[16]

这正是本文第 6 节的数学核心。

### C. Unified Bridge Algorithm：把 Flow Matching 和 Schrödinger Matching 放到一起

2025 年的 Unified Framework for Diffusion Bridge Problems 明确把 bridge problem 定义为找一个 SDE/ODE 连接两个分布，并提出 UBA 统一 Flow Matching、OT-CFM、Schrödinger bridge CFM 和 DSBM。[17]

这条线非常像 Brownian bridge / Schrödinger bridge 方向的“总框架”。

### D. Latent Stochastic Interpolants：最像“path-space VAE”的工作

Latent Stochastic Interpolants 把 stochastic interpolants 搬到 latent space，联合学习 encoder、decoder 和 latent SI model，并且直接从 continuous time 推导 ELBO。[20]

这可能是目前最接近“把 diffusion/flow 写成 learned latent graphical model”的论文之一。

### E. Field Matching 与 Flow/Field Matching duality：Poisson 路线正在被重新统一

2025 年的 Electrostatic Field Matching 把 source 和 target distribution 放在类似电容器的两块板上，以正负电荷产生 electrostatic field，用于生成和 distribution transfer。[18]

2026 年的 Flow/Field Matching duality 进一步指出：CFM 和 forward-only IFM 可以建立 bijection；但一般 IFM 更 expressive，并包含 EFM 等标准 CFM 不能直接表达的 interaction fields。[19]

这说明 Poisson/PFGM 不只是“物理灵感”，而可能是一个更大的 augmented-field generative model family。

### F. Schrödinger bridge as VAE

2024 年底的 Schödinger Bridge Type Diffusion Models as an Extension of Variational Autoencoders 明确尝试把 SB-type diffusion models 解释成 VAE 的扩展，并把目标函数拆成 prior loss 和 drift matching parts。[21]

这条线和本文的“diffusion 应被当作 VAE family”在哲学上非常一致。

---

## 12. 一个可能的研究纲领

如果我要继续推进这个方向，我会尝试定义这样一个模型类：

$$
\mathcal G
=
(
\Gamma_\phi,
R_\lambda,
\mathcal M,
a_\eta,
P_\theta
).
$$

其中：

- $\Gamma_\phi$：learned endpoint coupling / encoder coupling；
- $R_\lambda$：learned or designed bridge kernel；
- $\mathcal M$：state space，可以是 data space、latent space、manifold、augmented electrostatic space；
- $a_\eta$：stochastic gauge / diffusion matrix；
- $P_\theta$：causal decoder dynamics。

训练目标是：

$$
\min_{\theta,\phi,\lambda,\eta}
\mathrm{KL}
\left(
Q_{\phi,\lambda}(X_{0:1})
\| 
P_{\theta,\eta}(X_{0:1})
\right)
+
\text{endpoint reconstruction / likelihood term}.
$$

采样时：

1. 从 $z\sim\pi_0$ 开始；
2. 在 chosen gauge 下解 ODE/SDE；
3. 到达 data side；
4. 如果在 latent space，就再过 decoder。

这个框架里，很多看似不同的模型都会变成特殊情况：

| 模型 | Coupling | Bridge | Gauge | 学习目标 |
|---|---|---|---|---|
| DDPM / Score-SDE | Gaussian corruption | noisy diffusion bridge | SDE | score / denoising / ELBO |
| DDIM / probability flow | 同上 | 同上 | ODE gauge | deterministic sampling |
| Rectified Flow | independent/OT coupling | straight-line bridge | ODE | velocity regression |
| Flow Matching | conditional path | Gaussian/OT path | ODE | vector field regression |
| Stochastic Interpolants | arbitrary endpoint densities + latent variable | stochastic interpolant | ODE/SDE both | velocity/score objectives |
| Schrödinger Bridge | entropic endpoint/path coupling | Brownian reference bridge | SDE | path KL / drift matching |
| PFGM/PFGM++ | augmented prior/data geometry | electrostatic field line | ODE-like field | field matching |
| Latent SI | learned encoder posterior | latent stochastic interpolant | ODE/SDE | continuous-time ELBO |

---

## 13. 最后一句话

我认为 diffusion 和 rectified flow 的真正统一，不应该是“找一个公式把 SDE 变成 ODE”这么简单。真正的统一应该是：

$$
\boxed{
\text{生成模型是在学习一条 latent path 的图模型。}
}
$$

Brownian bridge 解决的是 path entropy / reference process；Poisson flow 解决的是 augmented-space field geometry；Rectified Flow 解决的是 causal Markov projection；Diffusion 解决的是 noisy gauge 和 score learning；VAE 解决的是 encoder-decoder latent graphical model。

把它们放在一起，最自然的名字就是：

$$
\boxed{\textbf{Path-space VAE with probability-current gauge}.}
$$

我觉得这比单纯说“diffusion 和 flow 是等价的”更接近本质。

---

# 参考文献与继续阅读

[1] Ho, Jain, Abbeel. **Denoising Diffusion Probabilistic Models**. arXiv:2006.11239, 2020. <https://arxiv.org/abs/2006.11239>

[2] Kingma, Salimans, Poole, Ho. **Variational Diffusion Models**. arXiv:2107.00630 / NeurIPS 2021. <https://arxiv.org/abs/2107.00630>

[3] Huang et al. **A Variational Perspective on Diffusion-Based Generative Models and Score Matching**. NeurIPS 2021. <https://openreview.net/forum?id=bXehDYUjjXi>

[4] Song et al. **Score-Based Generative Modeling through Stochastic Differential Equations**. arXiv:2011.13456 / ICLR 2021. <https://arxiv.org/abs/2011.13456>

[5] Liu, Gong, Liu. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**. arXiv:2209.03003, 2022. <https://arxiv.org/abs/2209.03003>

[6] Q. Liu. **Rectified Flow: A Marginal Preserving Approach to Optimal Transport**. arXiv:2209.14577, 2022. <https://arxiv.org/abs/2209.14577>

[7] Lipman et al. **Flow Matching for Generative Modeling**. arXiv:2210.02747 / ICLR 2023. <https://arxiv.org/abs/2210.02747>

[8] Tong et al. **Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport**. arXiv:2302.00482 / TMLR 2024. <https://arxiv.org/abs/2302.00482>

[9] Albergo, Boffi, Vanden-Eijnden. **Stochastic Interpolants: A Unifying Framework for Flows and Diffusions**. JMLR 2025. <https://www.jmlr.org/papers/v26/23-1605.html>

[10] Xu, Liu, Tegmark, Jaakkola. **Poisson Flow Generative Models**. arXiv:2209.11178 / NeurIPS 2022. <https://arxiv.org/abs/2209.11178>

[11] Xu et al. **PFGM++: Unlocking the Potential of Physics-Inspired Generative Models**. arXiv:2302.04265 / ICML 2023. <https://arxiv.org/abs/2302.04265>

[12] De Bortoli et al. **Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling**. arXiv:2106.01357 / NeurIPS 2021. <https://arxiv.org/abs/2106.01357>

[13] Shi et al. **Diffusion Schrödinger Bridge Matching**. arXiv:2303.16852 / NeurIPS 2023. <https://arxiv.org/abs/2303.16852>

[14] Tong et al. **Simulation-Free Schrödinger Bridges via Score and Flow Matching**. arXiv:2307.03672 / PMLR 2024. <https://arxiv.org/abs/2307.03672>

[15] Liu et al. **Generalized Schrödinger Bridge Matching**. arXiv:2310.02233 / ICLR 2024. <https://arxiv.org/abs/2310.02233>

[16] Silveri, Conforti, Durmus. **Theoretical Guarantees in KL for Diffusion Flow Matching**. arXiv:2409.08311, 2024. <https://arxiv.org/abs/2409.08311>

[17] Kim. **A Unified Framework for Diffusion Bridge Problems: Flow Matching and Schrödinger Matching into One**. arXiv:2503.21756, 2025. <https://arxiv.org/abs/2503.21756>

[18] Kolesov et al. **Field Matching: an Electrostatic Paradigm to Generate and Transfer Data**. arXiv:2502.02367, 2025. <https://arxiv.org/abs/2502.02367>

[19] Shlenskii et al. **Unlocking the Duality between Flow and Field Matching**. arXiv:2602.02261, 2026. <https://arxiv.org/abs/2602.02261>

[20] Singh, Lagun. **Latent Stochastic Interpolants**. arXiv:2506.02276, accepted at ICLR 2026. <https://arxiv.org/abs/2506.02276>

[21] Kaba, Shimizu, Ohzeki, Sughiyama. **Schödinger Bridge Type Diffusion Models as an Extension of Variational Autoencoders**. arXiv:2412.18237, 2024. <https://arxiv.org/abs/2412.18237>
