# ICL-Based Conditional Generation for QM9: Complete Implementation Plan

**更新时间**: 2026-04-14  
**目标**: 基于 In-Context Learning 实现准确的分子性质条件生成，从 QM9 数据集开始验证

---

## 1. 核心设计理念

### 1.1 与传统条件生成的区别

**传统方式**:
```
训练: G(property_embedding) → molecule
推理: G(target_property) → molecule
```

**ICL 方式**:
```
训练: G(context_examples, target_property) → molecule
推理: G(few_shot_examples, target_property) → molecule
```

### 1.2 为什么 ICL 更适合精确条件生成

1. **显式展示 property-structure 映射**: 通过示例直接展示"这个性质对应这个结构"
2. **灵活的条件控制**: 可以通过选择不同的 ICL 样本来引导生成方向
3. **无需重训练**: 新的性质组合可以通过组合不同示例实现
4. **可解释性**: 生成结果可以追溯到参考的 ICL 样本

---

## 2. QM9 数据集分析

### 2.1 QM9 基本信息

- **规模**: ~134k 个小分子（最多 9 个重原子，C/H/O/N/F）
- **性质**: 19 个量子化学性质
- **常用性质**:
  - `mu`: 偶极矩 (Dipole moment)
  - `alpha`: 极化率 (Polarizability)
  - `homo`: HOMO 能量
  - `lumo`: LUMO 能量
  - `gap`: HOMO-LUMO gap
  - `r2`: 电子空间范围
  - `zpve`: 零点振动能
  - `U0`, `U`, `H`, `G`: 热力学性质
  - `Cv`: 热容

### 2.2 选择起步性质

**第一阶段建议**: 单性质条件生成
- 优先选择: `gap` (HOMO-LUMO gap)
  - 原因: 物理意义明确，与分子稳定性直接相关，分布相对连续

**第二阶段**: 双性质条件生成
- 组合: `gap` + `mu` (偶极矩)
  - 原因: 两者相对独立，可以验证多性质控制能力

---

## 3. 方法设计: ICL-MolDiT

### 3.1 整体架构

```
Input Context:
┌─────────────────────────────────────────┐
│ ICL Example 1: (mol_1, prop_1)         │
│ ICL Example 2: (mol_2, prop_2)         │
│ ...                                      │
│ ICL Example K: (mol_K, prop_K)         │
│ Target Query:  (?, target_prop)         │
└─────────────────────────────────────────┘
           ↓
    Graph DiT Backbone
           ↓
    Generated Molecule
```

### 3.2 输入表示设计

#### 方案 A: Token Sequence (推荐用于第一版)

```python
# 每个 ICL 样本的表示
example_tokens = [
    [EXAMPLE_START],
    [PROP_TOKEN, prop_value_discretized],  # 性质 token
    [MOL_START],
    node_features,  # 原子类型、位置等
    edge_features,  # 键类型
    [MOL_END],
    [EXAMPLE_END]
]

# 完整输入
input_sequence = [
    *example_1_tokens,
    *example_2_tokens,
    ...
    *example_k_tokens,
    [QUERY_START],
    [PROP_TOKEN, target_prop_value],
    [MOL_START],
    # 这里是要生成的分子（训练时有监督，推理时逐步去噪）
]
```

#### 方案 B: Graph-Level Context (更适合 Graph DiT)

```python
# 使用 cross-attention 机制
context_graphs = [
    (G_1, prop_1),
    (G_2, prop_2),
    ...
    (G_K, prop_K)
]

# 编码 context
context_embeddings = ContextEncoder(context_graphs)

# 生成时 cross-attend
query_graph = GraphDiT(
    noisy_graph,
    target_prop,
    context=context_embeddings  # cross-attention
)
```

### 3.3 模型架构细节

```python
class ICLMolDiT(nn.Module):
    def __init__(self):
        # 1. Context Encoder
        self.context_encoder = GraphTransformer(
            num_layers=6,
            hidden_dim=256,
            num_heads=8
        )
        
        # 2. Property Encoder
        self.prop_encoder = nn.Sequential(
            nn.Linear(num_properties, 128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )
        
        # 3. Denoising Backbone (Graph DiT)
        self.denoiser = GraphDiTBlock(
            num_layers=12,
            hidden_dim=256,
            num_heads=8,
            use_cross_attention=True  # 用于 attend context
        )
        
        # 4. Output Heads
        self.node_head = nn.Linear(256, num_atom_types)
        self.edge_head = nn.Linear(256, num_bond_types)
    
    def forward(self, noisy_graph, target_prop, context_examples, timestep):
        # Encode context examples
        context_emb = []
        for mol, prop in context_examples:
            mol_emb = self.context_encoder(mol)
            prop_emb = self.prop_encoder(prop)
            context_emb.append(mol_emb + prop_emb)
        context_emb = torch.stack(context_emb)  # [K, N, D]
        
        # Encode target property
        target_prop_emb = self.prop_encoder(target_prop)
        
        # Denoise with context
        denoised = self.denoiser(
            noisy_graph,
            timestep,
            condition=target_prop_emb,
            context=context_emb  # cross-attention
        )
        
        # Predict clean graph
        node_pred = self.node_head(denoised.node_features)
        edge_pred = self.edge_head(denoised.edge_features)
        
        return node_pred, edge_pred
```

---

## 4. 训练策略

### 4.1 数据准备

#### Step 1: 性质归一化
```python
# 对每个性质做标准化
property_stats = {
    'gap': {'mean': 5.0, 'std': 2.5},
    'mu': {'mean': 2.0, 'std': 1.5},
    ...
}

def normalize_property(prop_value, prop_name):
    stats = property_stats[prop_name]
    return (prop_value - stats['mean']) / stats['std']
```

#### Step 2: ICL 样本采样策略

**策略 A: Random Sampling** (Baseline)
```python
def sample_icl_examples(target_prop, dataset, K=3):
    # 随机采样 K 个样本
    return random.sample(dataset, K)
```

**策略 B: Property-Guided Sampling** (推荐)
```python
def sample_icl_examples(target_prop, dataset, K=3):
    # 在目标性质附近采样
    # 例如: gap_target = 5.0, 采样 gap ∈ [4.5, 5.5] 的分子
    candidates = [
        mol for mol in dataset
        if abs(mol.property - target_prop) < threshold
    ]
    
    # 保证多样性: 在候选中均匀采样
    return stratified_sample(candidates, K)
```

**策略 C: Curriculum Sampling** (进阶)
```python
def sample_icl_examples(target_prop, dataset, K=3, epoch):
    # 早期: 采样非常接近的样本（简单）
    # 后期: 采样更远的样本（困难）
    threshold = initial_threshold * decay_factor ** epoch
    ...
```

### 4.2 训练目标

#### 主损失: Denoising Objective
```python
# 标准 diffusion 损失
def diffusion_loss(model, clean_graph, target_prop, context_examples):
    # 1. 加噪
    t = sample_timestep()
    noisy_graph = add_noise(clean_graph, t)
    
    # 2. 预测
    pred_clean = model(noisy_graph, target_prop, context_examples, t)
    
    # 3. 计算损失
    loss = MSE(pred_clean, clean_graph)
    return loss
```

#### 辅助损失: Property Prediction (可选)
```python
# 让模型学会从生成的分子预测性质
def property_prediction_loss(model, generated_graph, true_prop):
    # 使用独立的 property predictor
    pred_prop = property_predictor(generated_graph)
    loss = MSE(pred_prop, true_prop)
    return loss

# 总损失
total_loss = diffusion_loss + λ * property_prediction_loss
```

**注意**: 根据你的要求，property predictor 仅用于评估，不参与生成器训练。所以这个辅助损失是**可选的**，如果加的话权重 λ 要很小。

### 4.3 训练流程

```python
for epoch in epochs:
    for batch in dataloader:
        # 1. 采样 ICL 样本
        context_examples = [
            sample_icl_examples(mol.property, dataset, K=3)
            for mol in batch
        ]
        
        # 2. 前向传播
        loss = diffusion_loss(
            model,
            clean_graph=batch.graphs,
            target_prop=batch.properties,
            context_examples=context_examples
        )
        
        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4.4 训练技巧

1. **Classifier-Free Guidance 训练**
```python
# 随机丢弃 context (10-20% 概率)
if random.random() < 0.15:
    context_examples = []  # 无条件生成
```

2. **Context Dropout**
```python
# 随机丢弃部分 ICL 样本
K_actual = random.randint(1, K_max)
context_examples = context_examples[:K_actual]
```

3. **Property Noise Augmentation**
```python
# 给 ICL 样本的性质加噪声，增强鲁棒性
noisy_prop = true_prop + noise * std
```

---

## 5. 推理策略

### 5.1 基础推理

```python
def generate(model, target_prop, icl_examples, num_steps=1000):
    # 1. 初始化随机图
    graph = initialize_random_graph()
    
    # 2. 逆扩散
    for t in reversed(range(num_steps)):
        # 预测去噪方向
        pred = model(graph, target_prop, icl_examples, t)
        
        # 更新图
        graph = diffusion_step(graph, pred, t)
    
    return graph
```

### 5.2 ICL 样本选择策略

**策略 1: Nearest Neighbor Retrieval**
```python
def select_icl_examples(target_prop, dataset, K=3):
    # 找到性质最接近的 K 个分子
    distances = [abs(mol.property - target_prop) for mol in dataset]
    indices = np.argsort(distances)[:K]
    return [dataset[i] for i in indices]
```

**策略 2: Diverse Retrieval**
```python
def select_icl_examples(target_prop, dataset, K=3):
    # 在目标性质附近找到结构多样的样本
    candidates = filter_by_property_range(dataset, target_prop, threshold)
    
    # 最大化结构多样性
    selected = []
    for _ in range(K):
        if not selected:
            selected.append(random.choice(candidates))
        else:
            # 选择与已选样本最不相似的
            scores = [
                min(structure_similarity(mol, s) for s in selected)
                for mol in candidates
            ]
            selected.append(candidates[np.argmin(scores)])
    
    return selected
```

**策略 3: Interpolation Chain**
```python
def select_icl_examples(target_prop, dataset, K=3):
    # 构造一个从低到高的性质链
    # 例如: gap=3.0 → gap=4.0 → gap=5.0 (target)
    prop_range = np.linspace(
        target_prop - delta,
        target_prop,
        K
    )
    return [find_nearest(dataset, p) for p in prop_range]
```

### 5.3 Classifier-Free Guidance

```python
def generate_with_cfg(model, target_prop, icl_examples, guidance_scale=2.0):
    for t in reversed(range(num_steps)):
        # 有条件预测
        pred_cond = model(graph, target_prop, icl_examples, t)
        
        # 无条件预测
        pred_uncond = model(graph, target_prop, [], t)
        
        # 组合
        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        
        graph = diffusion_step(graph, pred, t)
    
    return graph
```

---

## 6. 评估指标

### 6.1 生成质量

1. **Validity**: 生成分子的化学合法性
   ```python
   validity = sum(is_valid(mol) for mol in generated) / len(generated)
   ```

2. **Uniqueness**: 去重后的比例
   ```python
   uniqueness = len(set(generated)) / len(generated)
   ```

3. **Novelty**: 不在训练集中的比例
   ```python
   novelty = sum(mol not in train_set for mol in generated) / len(generated)
   ```

### 6.2 条件控制精度 (核心指标)

1. **Property MAE**: 生成分子的性质与目标的平均绝对误差
   ```python
   mae = mean(|predictor(mol) - target_prop| for mol in generated)
   ```

2. **Property Hit Rate**: 在容忍范围内的比例
   ```python
   hit_rate = sum(
       |predictor(mol) - target_prop| < threshold
       for mol in generated
   ) / len(generated)
   ```

3. **Property Distribution Alignment**: KL 散度或 Wasserstein 距离
   ```python
   kl_div = KL(P_generated || P_target)
   ```

### 6.3 ICL 效果评估

1. **Context Sensitivity**: 改变 ICL 样本后生成结果的变化
   ```python
   # 用不同 ICL 样本生成，看结果是否合理变化
   diversity = mean_pairwise_distance(
       [generate(target_prop, icl_set_i) for icl_set_i in different_contexts]
   )
   ```

2. **Ablation**: 比较不同 K 值的效果
   ```python
   for K in [0, 1, 3, 5, 10]:
       results[K] = evaluate(model, K_examples=K)
   ```

---

## 7. 实验设计

### 7.1 Baseline 对比

1. **Unconditional Generation**: 无条件生成 + 后筛选
2. **Simple Conditional**: 直接用 property embedding 作为条件（无 ICL）
3. **Classifier Guidance**: 用预训练 property predictor 引导采样
4. **Best-of-N**: 生成 N 个样本，选择性质最接近的

### 7.2 消融实验

| 实验组 | ICL | Property Condition | CFG | 说明 |
|--------|-----|-------------------|-----|------|
| A | ✗ | ✗ | ✗ | 无条件生成 |
| B | ✗ | ✓ | ✗ | 传统条件生成 |
| C | ✓ | ✗ | ✗ | 仅 ICL，无显式性质条件 |
| D | ✓ | ✓ | ✗ | ICL + 性质条件 |
| E | ✓ | ✓ | ✓ | 完整方法 |

### 7.3 ICL 样本数量实验

```python
K_values = [1, 3, 5, 10, 20]
for K in K_values:
    mae, hit_rate = evaluate(model, K_examples=K)
    plot(K, mae, hit_rate)
```

### 7.4 ICL 采样策略对比

- Random sampling
- Property-guided sampling
- Diverse retrieval
- Interpolation chain

---

## 8. 进阶: Reflect-DiT 式 Refinement

在基础 ICL 生成成功后，可以加入 reflection loop:

### 8.1 Refinement 流程

```python
def generate_with_refinement(model, target_prop, icl_examples, max_rounds=3):
    # Round 0: 初始生成
    mol_0 = generate(model, target_prop, icl_examples)
    
    history = []
    for round in range(max_rounds):
        # 1. 评估当前分子
        pred_prop = property_predictor(mol_t)
        error = target_prop - pred_prop
        
        # 2. 如果满足条件，提前停止
        if abs(error) < threshold:
            return mol_t
        
        # 3. 构造反馈
        feedback = {
            'property_error': error,
            'direction': 'increase' if error > 0 else 'decrease',
            'magnitude': abs(error)
        }
        
        # 4. 更新历史
        history.append((mol_t, feedback))
        
        # 5. 带历史重新生成
        mol_t_plus_1 = generate(
            model,
            target_prop,
            icl_examples,
            history=history  # 新增历史上下文
        )
        
        mol_t = mol_t_plus_1
    
    return mol_t
```

### 8.2 History Encoding

```python
class ICLMolDiTWithHistory(ICLMolDiT):
    def __init__(self):
        super().__init__()
        self.history_encoder = HistoryEncoder(hidden_dim=256)
    
    def forward(self, noisy_graph, target_prop, context_examples, 
                timestep, history=None):
        # 编码 context
        context_emb = self.encode_context(context_examples)
        
        # 编码 history
        if history:
            history_emb = self.history_encoder(history)
            # 拼接或相加
            context_emb = torch.cat([context_emb, history_emb], dim=0)
        
        # 其余同前
        ...
```

---

## 9. 实施路线图

### Phase 1: 基础 ICL 生成 (2-3 周)

**Week 1: 数据 + Baseline**
- [ ] 下载并预处理 QM9 数据集
- [ ] 实现数据加载器（支持 ICL 样本采样）
- [ ] 训练一个简单的无条件 Graph DiT baseline
- [ ] 训练一个 property predictor（用于评估）

**Week 2: ICL 模型**
- [ ] 实现 ICL-MolDiT 架构
- [ ] 实现 context encoder + cross-attention
- [ ] 训练 ICL 模型（单性质: gap）
- [ ] 实现 CFG 推理

**Week 3: 评估 + 消融**
- [ ] 实现评估指标（validity, MAE, hit rate）
- [ ] 对比 baseline（无条件、简单条件、ICL）
- [ ] 消融实验（K 值、采样策略）
- [ ] 可视化分析

### Phase 2: 多性质 + Refinement (2-3 周)

**Week 4: 多性质控制**
- [ ] 扩展到双性质（gap + mu）
- [ ] 实现多性质 ICL 样本选择
- [ ] 评估多性质控制精度

**Week 5: Reflection Loop**
- [ ] 实现 history encoder
- [ ] 实现 refinement 推理流程
- [ ] 构造 refinement 训练数据（self-play）
- [ ] 训练 refinement 模型

**Week 6: 完整评估**
- [ ] 对比 refinement vs. best-of-N
- [ ] 分析每轮 refinement 的改进幅度
- [ ] 撰写技术报告

### Phase 3: 迁移到晶体 (后续)

- [ ] 适配 CrystalDiT backbone
- [ ] 设计晶体的 ICL 表示
- [ ] 实现晶体 property predictor
- [ ] 完整流程验证

---

## 10. 技术风险与缓解

### 风险 1: ICL 样本选择不当导致生成失败
**缓解**:
- 实现多种采样策略并对比
- 加入 context dropout 增强鲁棒性
- 可视化 ICL 样本对生成结果的影响

### 风险 2: Property predictor 不准确导致评估偏差
**缓解**:
- 使用多个独立训练的 predictor 做 ensemble
- 对部分生成分子做 DFT 验证
- 报告 predictor 的置信区间

### 风险 3: 模型只是记忆 ICL 样本，而非学习映射
**缓解**:
- 测试时用训练集外的 ICL 样本
- 检查生成分子与 ICL 样本的结构相似度
- 做 interpolation 实验（目标性质在 ICL 样本之间）

### 风险 4: 小分子空间有限，难以体现 ICL 优势
**缓解**:
- 选择性质分布广的子集
- 设计更困难的条件（多性质、稀有区域）
- 如果 QM9 太简单，快速迁移到 ZINC 或 GEOM

---

## 11. 预期成果

### 11.1 技术贡献

1. **首个系统性的 ICL 分子生成方法**: 明确的 ICL 样本设计 + 训练策略
2. **更精确的条件控制**: 相比传统 property embedding，ICL 提供更强的引导
3. **可扩展框架**: 从 QM9 到晶体的通用 pipeline

### 11.2 实验结果预期

| 指标 | Baseline | ICL (K=3) | ICL + Refinement |
|------|----------|-----------|------------------|
| Property MAE | 1.5 | 0.8 | 0.4 |
| Hit Rate@0.5 | 30% | 60% | 80% |
| Validity | 95% | 95% | 95% |

### 11.3 论文方向

**标题候选**:
- "In-Context Learning for Precise Property-Conditioned Molecular Generation"
- "ICL-MolDiT: Few-Shot Molecular Design via Diffusion Transformers"
- "Learning Property-Structure Mappings through In-Context Molecular Generation"

**投稿目标**: NeurIPS, ICML, ICLR (AI4Science track)

---

## 12. 代码结构建议

```
icl-moldit/
├── data/
│   ├── qm9.py              # QM9 数据加载
│   ├── icl_sampler.py      # ICL 样本采样策略
│   └── preprocess.py       # 数据预处理
├── models/
│   ├── graph_dit.py        # Graph DiT backbone
│   ├── icl_encoder.py      # Context encoder
│   ├── icl_moldit.py       # 完整 ICL 模型
│   └── property_predictor.py
├── training/
│   ├── train.py            # 训练脚本
│   ├── losses.py           # 损失函数
│   └── diffusion.py        # 扩散过程
├── inference/
│   ├── generate.py         # 基础生成
│   ├── refinement.py       # Reflection loop
│   └── icl_selection.py    # ICL 样本选择
├── evaluation/
│   ├── metrics.py          # 评估指标
│   ├── visualize.py        # 可视化
│   └── analyze.py          # 结果分析
└── configs/
    ├── base.yaml           # 基础配置
    └── icl.yaml            # ICL 配置
```

---

## 13. 下一步行动

1. **立即开始**: 下载 QM9 数据集，熟悉数据格式
2. **文献精读**: 仔细阅读 Graph-DiT 和 Reflect-DiT 论文
3. **代码调研**: 找 Graph-DiT 的开源实现作为 backbone
4. **原型实验**: 先用最简单的设置（K=1, 单性质）验证可行性

需要我帮你开始实现某个具体部分吗？
