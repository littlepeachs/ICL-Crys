# 将 Reflect-DiT 式 Refine 迁移到 CrystalDiT / Graph-DiT 的研究文档

更新时间：2026-04-08  
目的：评估是否已有“类似 Reflect-DiT 的 verifier-reflection-refine”工作，并给出把该模式迁移到晶体生成和分子图生成的可执行方案。

## 1. 先说结论

基于本次网络检索，我**没有检索到一篇已经完整实现**下列闭环的晶体/分子扩散工作：

1. 先生成一个候选结构；
2. 用外部 verifier 或 critic 产生可解释反馈；
3. 把“历史样本 + 历史反馈”重新送回生成器；
4. 多轮迭代 refined generation；
5. 在 test-time 明确获得随迭代上升的 success rate。

这个结论是**基于我本次检索结果的归纳**，不是对全部文献的绝对证明。  
我找到的最接近工作分成三类：

1. `sampling-time guidance`：在采样过程中用 predictor / gradient / guidance 场引导生成，但没有显式“历史错误记忆”。
2. `RL / reward fine-tuning`：把 property 或稳定性作为奖励微调生成器，但通常不是 Reflect-DiT 那种 test-time reflection loop。
3. `self-criticism / editing`：分子侧已经出现了“自批评”或“编辑”方向，离 Reflect-DiT 最近，但多数仍停在 best-of-k 选择、单步编辑或 RL 编辑，没有显式多轮历史上下文。

因此，**把 Reflect-DiT 式 refine 迁到 CrystalDiT / Graph-DiT 是有研究空位的**。  
其中：

1. `CrystalDiT` 更适合先做“结构化 verifier + 重新生成”版本。
2. `Graph-DiT` 更适合同时探索“从头重采样 refine”和“基于当前图做局部编辑 refine”两条线。

## 2. Reflect-DiT 的可迁移核心

Reflect-DiT 的关键不是图像领域本身，而是下面这个抽象：

1. 有一个 generator `G`
2. 有一个 judge / verifier `J`
3. `J` 能把失败样本转成可用反馈 `f`
4. `G` 能接收 `(条件, 历史样本, 历史反馈)` 再生成下一个候选

抽象成统一形式：

`x_0 = G(c)`  
`f_t = J(c, x_t)`  
`x_{t+1} = G(c, H_t)`，其中 `H_t = {(x_i, f_i)}_{i=0}^t`

这里的 `c`：

1. 对晶体是组成、空间群、目标性质、元素约束、稳定性目标等；
2. 对分子是目标性质、骨架约束、子结构约束、可合成性目标、对接分数目标等。

只要我们能定义一个足够可靠的 verifier，这个框架就能从图像迁到材料或分子。

## 3. 检索到的最相关文献

### 3.1 图像侧母体

1. Reflect-DiT, ICCV 2025  
   链接：https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Reflect-DiT_Inference-Time_Scaling_for_Text-to-Image_Diffusion_Transformers_via_In-Context_Reflection_ICCV_2025_paper.pdf  
   相关性：直接提供了“生成 -> 反馈 -> 带历史再生成”的模板。

### 3.2 分子 / 图生成侧

1. Graph Diffusion Transformers for Multi-Conditional Molecular Generation, NeurIPS 2024  
   链接：https://openreview.net/forum?id=cfrDLD1wfO  
   要点：Graph-DiT 已证明“Transformer graph denoiser + 多属性条件控制”在聚合物和小分子上可行；摘要还提到一个“带 domain expert feedback 的 polymer inverse design”场景，但不是 learned reflection loop。  
   结论：这是迁移到分子图生成的最直接 backbone。

2. Any-Property-Conditional Molecule Generation with Self-Criticism using Spanning Trees, TMLR 2025  
   链接：https://openreview.net/forum?id=QGZd5Bfb1L  
   要点：STGG+ 引入 auxiliary property-prediction loss，让模型能“self-criticize molecules and select the best ones”。  
   结论：这是分子侧和 Reflect-DiT 最像的一条线，但它更像“自评 + 选优”，不是“记住错误轨迹再下一轮生成”。

3. Guided diffusion for inverse molecular design, Nature Computational Science 2023  
   链接：https://www.nature.com/articles/s43588-023-00532-0  
   要点：GaUDI 用性质预测器梯度直接引导逆扩散，做多目标分子设计。  
   结论：它说明分子侧的 verifier/guidance 完全可用，但它是连续 guidance，不是 reflection memory。

4. Exploring Chemical Space with Score-based Out-of-distribution Generation, ICML 2023  
   链接：https://openreview.net/forum?id=WP07wAWxty  
   要点：MOOD 在逆扩散时用 property predictor gradient 把采样推向高分区域。  
   结论：属于 sampling-time guidance，与 Reflect-DiT 互补，可作为强 baseline。

5. MolEditRL: Structure-Preserving Molecular Editing via Discrete Diffusion and Reinforcement Learning, ICLR 2026  
   链接：https://openreview.net/forum?id=40QphlZ9fY  
   要点：从给定源分子出发，在结构保持约束下做离散图编辑，并用 RL 强化性质对齐。  
   结论：如果 Graph-DiT 的 refine 不想“每轮都从头生成”，MolEditRL 代表了“局部编辑 refine”的更近邻方案。

### 3.3 晶体 / 材料生成侧

1. Crystal Structure Prediction by Joint Equivariant Diffusion, NeurIPS 2023  
   链接：https://openreview.net/forum?id=DNdN26m2Jk  
   要点：DiffCSP 奠定了晶体扩散生成里对 lattice + fractional coordinates 联合建模的标准路线。  
   结论：这是晶体 refine 的重要底座，但它本身没有 feedback loop。

2. A generative model for inorganic materials design, Nature 2025  
   链接：https://www.nature.com/articles/s41586-025-08628-5  
   要点：MatterGen 能生成稳定、novel、diverse 的无机材料，并可 fine-tune 到 chemistry / symmetry / scalar property constraints。  
   结论：说明材料侧“多属性条件生成 + adapter fine-tuning”已经成熟，但还是单次条件生成范式。

3. Reinforcement learning with formation energy feedback for material diffusion models, Neural Networks 2026  
   链接：https://www.sciencedirect.com/science/article/pii/S0893608025010263  
   要点：RLFEF 把 material diffusion process 视作 MDP，用 formation energy 作为 reward，对材料 diffusion 模型做 fine-tuning。  
   结论：这是材料侧最接近“反馈”二字的工作，但反馈进入的是训练期 RL，而不是 Reflect-DiT 式 test-time reflection。

4. MatInvent: Reinforcement Learning for 3D Crystal Diffusion Generation, AI4Mat-ICLR 2025  
   链接：https://openreview.net/forum?id=Ovxfri7l5L  
   要点：把晶体 diffusion 生成做成 goal-directed RL，加入 reward-weighted KL regularization、experience replay 和 diversity filter。  
   结论：它说明“外部性质目标 -> 生成器迭代优化”是可行的，但仍主要是 policy optimization，不是显式历史反馈上下文。

5. Train Separately, Compose at Sampling: Multi-Property Crystal Generation with Orthogonal Flow Guidance, AI4Mat-ICLR 2026  
   链接：https://openreview.net/forum?id=PjkeOpfo8c  
   要点：先学单性质 guidance，再在采样时组合 guidance fields。  
   结论：这代表了材料侧另一类强 baseline：sampling-time guidance，而非 reflection loop。

## 4. 对现有文献的判断

### 4.1 哪些工作“真的像” Reflect-DiT

最像的有两篇：

1. `STGG+`
2. `MolEditRL`

但它们分别只覆盖了 Reflect-DiT 思想的一部分：

1. `STGG+` 有 self-criticism，但核心更偏“自评 + 选优”，还没有显式多轮历史上下文。
2. `MolEditRL` 有“基于当前分子修改”的编辑思路，但更像 instruction-conditioned editing + RL，不是 verifier-reflection trajectory。

### 4.2 哪些工作只是“邻近”

1. `GaUDI`
2. `MOOD`
3. `MatterGen`
4. `RLFEF`
5. `MatInvent`
6. `Orthogonal Flow Guidance`

这些工作都说明：

1. 外部性质预测器能稳定进入生成过程；
2. 采样时 guidance 和训练时 RL 都是可行的；
3. 但“把失败样本与错误解释存下来，再喂回模型”的路线还没有被系统做透。

这正是你可以切进去的点。

## 5. 迁移到 CrystalDiT 的推荐方案

### 5.1 先定问题形式

对晶体生成，我们定义：

1. 条件 `c`：组成、化学系统、空间群、目标 band gap、formation energy、magnetic density、bulk modulus 等；
2. 样本 `x`：晶体结构，包含原子类型、分数坐标、晶格参数；
3. verifier `J(c, x)`：输出结构是否合法、稳定性如何、哪些性质没达标；
4. generator `G(c, H_t)`：接收条件和历史反馈上下文，生成下一版晶体。

目标不是单纯最大化某个 reward，而是最大化：

1. 结构有效性；
2. 稳定性；
3. 条件命中率；
4. 每轮 refine 后的 success@t / pass@t。

### 5.2 v1 不建议直接上“自然语言反馈”

图像里用自然语言反馈很自然，但晶体里直接用自由文本反馈未必是最优。  
更稳的 v1 是**结构化 feedback tokens**。

推荐反馈 schema：

1. `composition_mismatch`
2. `space_group_mismatch`
3. `atom_count_invalid`
4. `bond_length_clash` 或 `min_distance_violation`
5. `charge_neutrality_violation`
6. `formation_energy_too_high`
7. `ehull_too_high`
8. `band_gap_too_low` / `band_gap_too_high`
9. `mag_density_too_low`
10. `relaxation_unstable`

每条反馈再带一个 residual：

1. 符号：高了还是低了
2. 幅度：差多少
3. 置信度：verifier 的可信程度

这样做比自由文本更容易训练，也更适合数值性质。

### 5.3 CrystalDiT 的上下文如何注入

建议把每轮历史组织成：

`(x_t, p_t, f_t)`

其中：

1. `x_t`：上一轮晶体结构编码
2. `p_t`：verifier 给出的性质估计向量
3. `f_t`：结构化 feedback token 序列

注入方式有三种，从易到难：

1. `late fusion condition tokens`
   把 `f_t` 和 `p_t` 编成额外条件 token，和原条件一起送入 DiT。
2. `history memory encoder`
   用一个 periodic GNN / crystal encoder 把 `x_t` 编成 memory，再与 feedback token 拼接后 cross-attend。
3. `edit-state denoising`
   让模型不仅看条件，还看“当前候选结构”的 noisy 版本，直接学局部修正。

如果是第一版，我建议优先做 `1 + 2`，先不要急着做真正的 edit diffusion。

### 5.4 训练数据怎么来

这是整个方案最关键的难点。图像侧有坏图和好图的配对逻辑，但晶体侧未必天然有“坏晶体 -> 对应修正版晶体”。

可以用四种方式构造 trajectory：

1. `offline self-play`
   用现有 CrystalDiT 按条件生成多个候选，跑 verifier，保留每轮改进轨迹，构造 `(x_t, f_t, x_{t+1})` 伪标签。
2. `retrieval teacher`
   对于一个失败样本 `x_t`，在真实数据集中检索满足条件的最近邻好样本 `x*`，训练模型学会“看着失败原因去靠近可行区”。
3. `relax-and-relabel`
   对候选结构做快速几何优化、MLIP 松弛或启发式修复，把修复后结构视为改进目标。
4. `best-of-k bootstrapping`
   每轮采样 `k` 个候选，用 verifier 选最优作为下一轮监督信号，逐步蒸馏出 refine 行为。

最现实的起点是：

1. 先做 `offline self-play`
2. 再加 `best-of-k bootstrapping`

因为这两条最不依赖昂贵标注。

### 5.5 verifier 体系建议分层

不要一开始就全靠 DFT。  
建议三层 verifier：

1. `cheap hard checks`
   组成合法性、原子间最短距离、价态/电荷近似规则、空间群约束。
2. `surrogate predictors`
   formation energy、band gap、磁性、力学性质，用预训练 GNN / MLP predictor。
3. `expensive oracle`
   少量 DFT / relaxation，只用于高价值样本和后验评估。

这样才能把反思 loop 的开销压住。

### 5.6 CrystalDiT 的最小研究版本

我建议按下面顺序做：

1. 不改 backbone，只在输入侧新增 `feedback tokens + previous property vector`
2. 先只保留 `1-step refinement`，即 `x_0 -> f_0 -> x_1`
3. 只用 surrogate verifier
4. 只优化一个主性质加一个稳定性指标
5. 和以下 baseline 比：
   - base CrystalDiT
   - best-of-N
   - predictor guidance
   - RL fine-tuning

如果 `1-step refinement` 已经稳定胜过 `best-of-N`，再扩展到多轮历史。

## 6. 迁移到 Graph-DiT 的推荐方案

### 6.1 Graph-DiT 比 CrystalDiT 更容易做 refine

原因有三个：

1. 分子合法性检查更成熟，RDKit 规则可直接用；
2. 性质预测器更便宜；
3. “图编辑”本身就比晶体 lattice/坐标的局部修正更自然。

所以如果你的目标是先打出一个更容易发的原型，**Graph-DiT 线可能比 CrystalDiT 更快出结果**。

### 6.2 Graph-DiT 上的 feedback schema

建议分四类：

1. `validity feedback`
   价态非法、芳香性冲突、连通性错误、环结构异常。
2. `property residual feedback`
   QED 太低、SA 太差、logP 偏高、毒性风险高、结合分数不足。
3. `constraint feedback`
   scaffold 未保留、子结构未出现、官能团缺失。
4. `distribution feedback`
   与训练分布偏离过大、novelty 虽高但 plausibility 低。

这些都能写成离散 token + 数值 residual。

### 6.3 两条可并行研究路线

#### 路线 A：from-scratch refine

每一轮都重新采样一个新分子图：

`G_{t+1} = Model(c, history)`

优点：

1. 最接近 Reflect-DiT；
2. 实现改动小；
3. 不需要定义复杂图编辑算子。

缺点：

1. 可能浪费已接近正确的局部结构；
2. 多轮之间的结构连续性较差。

#### 路线 B：edit-based refine

每轮在当前图上做局部编辑：

1. 加/删节点
2. 改边类型
3. 子图替换
4. scaffold-preserving rewrite

优点：

1. 更符合“refine”直觉；
2. 更适合做 property-constrained optimization；
3. 更容易解释每轮修改了什么。

缺点：

1. 训练数据构造更复杂；
2. 编辑动作空间设计难。

如果你要做一条最稳妥路线：

1. 先做 `A`
2. 成功后再做 `B`

### 6.4 Graph-DiT 的训练构造

对每个目标条件 `c`：

1. 用 base Graph-DiT 采样多个候选；
2. 用 verifier 打分并生成反馈；
3. 取更好的候选作为 pseudo target；
4. 训练 `p(x_{t+1} | c, x_t, f_t)`；
5. 随机丢弃部分 feedback，增强鲁棒性；
6. 用 classifier-free guidance 风格训练“有反馈/无反馈”双模式。

可选增强：

1. 加一个 auxiliary critic head，直接预测“当前样本离满足目标还有多远”；
2. 在推理时混合 `self-critique score` 和外部 oracle score；
3. 用 `best-of-k within each round` 提高 trajectory 质量。

### 6.5 Graph-DiT 的最小研究版本

推荐先做：

1. 小分子，而不是先做 polymer
2. 单性质目标，例如 `QED` 或 `logP`
3. `1-step refine`
4. 结构化 feedback，不上自由文本
5. baseline:
   - Graph-DiT
   - Graph-DiT + best-of-N
   - Graph-DiT + predictor guidance
   - STGG+ style self-criticism selection

这条线很容易回答一个清晰问题：

“反馈驱动的第二次生成，是否比单纯多采样再筛选更有效？”

## 7. 一个统一的 Reflect-Gen 框架

### 7.1 通用算法

1. 输入条件 `c`
2. 初始生成 `x_0 ~ G(c)`
3. 用 verifier 得到 `f_0 = J(c, x_0)`
4. 若满足停止条件，则返回 `x_0`
5. 构造历史 `H_t = {(x_i, f_i)}`
6. 生成 `x_{t+1} ~ G(c, H_t)`
7. 重复直到达到 `N` 轮或 verifier 判定通过

### 7.2 停止条件

推荐不要只用“完全满足目标”作为 stop，而是并用：

1. 所有 hard constraints 满足
2. 主性质进入容忍区间
3. 连续两轮改进幅度小于阈值
4. verifier 置信度下降

## 8. 关键实验设计

### 8.1 你需要的核心问题

1. refine 是否优于 `best-of-N`？
2. refine 是否优于 `sampling-time guidance`？
3. refine 的收益来自“反馈本身”还是“只是多了一轮采样”？
4. 多轮历史是否优于单轮反馈？
5. 结构化 feedback 是否已经够用，自由文本是否真有额外价值？

### 8.2 建议指标

#### 晶体

1. validity
2. symmetry validity
3. SUN / novel-stable-unique 类指标
4. formation energy / `E_hull`
5. target property hit rate
6. pass@1, pass@2, pass@3
7. average improvement per iteration
8. oracle calls per successful sample

#### 分子

1. validity
2. uniqueness
3. novelty
4. property hit rate
5. top-k reward
6. SA / QED / docking proxy
7. pass@t
8. edit distance 或 scaffold retention

### 8.3 必做消融

1. 无历史，只给本轮反馈
2. 有历史，但不提供反馈文本/ token
3. 只给 property vector，不给离散错误标签
4. 单轮 refine vs 多轮 refine
5. from-scratch vs edit-based
6. structured feedback vs natural language feedback

## 9. 主要风险

### 9.1 verifier 噪声

如果 surrogate predictor 不准，模型会学会迎合 predictor，而不是真正生成更好的结构。

缓解：

1. 输出 verifier confidence
2. 多 verifier 集成
3. 少量高精度 oracle 做再标定

### 9.2 reward hacking / predictor hacking

模型可能生成“在 predictor 上高分、在真实物理上无意义”的结构。

缓解：

1. 加 hard validity checks
2. 加 distribution regularization
3. 保持与训练数据流形的距离约束

### 9.3 trajectory supervision 稀缺

坏样本到好样本的配对不天然存在。

缓解：

1. self-play bootstrapping
2. retrieval teacher
3. best-of-k pseudo target

### 9.4 多轮反思导致 mode collapse

模型可能越来越保守，只会往训练集高密度区缩。

缓解：

1. 每轮保留温度或噪声
2. 加 diversity bonus
3. 在 history 中混入失败样本而非只保留最佳样本

## 10. 我建议你怎么起步

### 10.1 如果要先做一个更容易成功的版本

优先顺序：

1. `Graph-DiT + structured feedback + 1-step refine`
2. `CrystalDiT + structured feedback + 1-step refine`
3. 再做多轮历史
4. 最后再尝试自然语言反馈

原因：

1. Graph verifier 更成熟；
2. 图编辑/图合法性更容易做；
3. 能更快验证“reflection 是否真比 best-of-N 强”。

### 10.2 如果你更想做材料侧的原创点

最值得发的一条线是：

`CrystalDiT + hierarchical verifier + feedback-conditioned resampling`

论文角度可以主打：

1. 首个把 Reflect-style refinement 系统迁到晶体生成；
2. 提出适合晶体的 structured feedback ontology；
3. 在固定 oracle budget 下优于 best-of-N、guidance、RL fine-tuning；
4. 支持多轮 pass@t 提升。

## 11. 一个可直接立项的题目

题目候选：

1. `ReflectCrystal: Verifier-Guided Iterative Refinement for Crystal Diffusion Transformers`
2. `ReflectGraphDiT: Feedback-Conditioned Iterative Refinement for Molecular Graph Diffusion`
3. `Structured Reflection for Inverse Materials Design`

一句话问题定义：

“给定目标性质与约束，能否让材料/分子生成模型在 test-time 通过结构化 verifier 反馈进行多轮自我修正，而不是仅依赖单次条件生成、采样 guidance 或 best-of-N 选择？”

## 12. 参考链接

1. Reflect-DiT: https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Reflect-DiT_Inference-Time_Scaling_for_Text-to-Image_Diffusion_Transformers_via_In-Context_Reflection_ICCV_2025_paper.pdf
2. Graph-DiT: https://openreview.net/forum?id=cfrDLD1wfO
3. STGG+: https://openreview.net/forum?id=QGZd5Bfb1L
4. GaUDI: https://www.nature.com/articles/s43588-023-00532-0
5. MOOD: https://openreview.net/forum?id=WP07wAWxty
6. MolEditRL: https://openreview.net/forum?id=40QphlZ9fY
7. DiffCSP: https://openreview.net/forum?id=DNdN26m2Jk
8. MatterGen: https://www.nature.com/articles/s41586-025-08628-5
9. RLFEF: https://www.sciencedirect.com/science/article/pii/S0893608025010263
10. MatInvent: https://openreview.net/forum?id=Ovxfri7l5L
11. Orthogonal Flow Guidance: https://openreview.net/forum?id=PjkeOpfo8c
