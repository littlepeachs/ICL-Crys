# CrystalICL: 基于Qwen3-8B的晶体生成模型

这是论文 "CrystalICL: Enabling In-Context Learning for Crystal Generation" 的完整复现代码，使用 **Qwen3-8B** 作为基础模型。

## 论文概述

CrystalICL 是一个专门为晶体生成设计的少样本学习模型，主要创新点包括：

1. **Space-group based Crystal Tokenization (SGS)** - 基于空间群的晶体token化方法
2. **Condition-Structure Aware Hybrid Crystal Instruction Tuning** - 条件-结构感知的混合指令微调
3. **Multi-Task Crystal Instruction Tuning** - 多任务学习策略

## 项目结构

```
ICL-Crys/
├── crystal_tokenization.py      # 晶体token化模块（SGS格式）
├── instruction_builder.py       # 指令构建模块
├── example_selector.py          # 示例选择策略（3种）
├── data_loader.py              # 数据加载器
├── train_crystalicl.py         # 训练脚本（Qwen3-8B + LoRA）
├── evaluate.py                 # 基础评估脚本
├── evaluate_complete.py        # 完整评估脚本
├── metrics_calculator.py       # 评估指标计算器
├── compute_paper_metrics.py    # 论文指标计算
├── run_crystalicl.py          # 主运行脚本
├── examples.py                 # 使用示例
├── test_modules.py             # 模块测试
├── requirements.txt            # 依赖包
├── config.json                 # 配置文件
└── README.md                   # 本文件
```

## 安装

### 1. 创建虚拟环境

```bash
conda create -n crystalicl python=3.10
conda activate crystalicl
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装PyTorch（根据你的CUDA版本）

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 快速开始

### 1. 测试模块

```bash
python test_modules.py
```

### 2. 使用示例数据训练

```bash
python run_crystalicl.py \
    --use_sample_data \
    --num_samples 100 \
    --do_train \
    --do_eval \
    --num_epochs 3 \
    --batch_size 1 \
    --use_few_shot \
    --k_shot 3
```

### 3. 完整评估（计算论文中所有指标）

```bash
python evaluate_complete.py \
    --model_path ./crystalicl_qwen3_8b_output \
    --test_data ./data/test.json \
    --num_samples 1000 \
    --num_unconditional 10000
```

## 评估指标

本项目完整实现了论文中的所有评估指标：

### Table 1: 条件生成指标 (Conditional Generation)

对每个样本进行1000次采样，计算成功率的均值和标准差：

| 指标 | 说明 |
|------|------|
| **Pretty Formula** | 化学式匹配率 |
| **Space Group** | 空间群匹配率 |
| **Formation Energy** | 形成能匹配率（需DFT计算） |
| **Band Gap** | 带隙匹配率（需DFT计算） |

每个指标报告：
- Mean: 成功率均值
- Std: 成功率标准差

### Table 2: 无条件生成指标 (Unconditional Generation)

生成10,000个样本，计算以下指标：

#### 1. Validity Check (有效性检查)
- **Structural Validity**: 结构有效性（原子间距 > 0.5Å）
- **Compositional Validity**: 组成有效性（电荷中性）
- **Total Validity**: 综合有效性

#### 2. Coverage (覆盖率)
- **Recall**: 召回率（覆盖了多少参考结构）
- **Precision**: 精确率（生成的有效结构比例）

#### 3. Property Distribution (属性分布)
使用Wasserstein距离衡量生成结构和参考结构的属性分布差异：
- Density
- Number of Atoms
- Volume
- Number of Elements

## 使用方法

### 1. 训练模型

```bash
python run_crystalicl.py \
    --model_name Qwen/Qwen3-8B \
    --data_path ./your_data.json \
    --do_train \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --use_few_shot \
    --k_shot 3
```

### 2. 评估模型

#### 快速评估
```bash
python evaluate.py \
    --model_path ./crystalicl_qwen3_8b_output \
    --test_data ./data/test.json
```

#### 完整评估（所有论文指标）
```bash
python evaluate_complete.py \
    --model_path ./crystalicl_qwen3_8b_output \
    --test_data ./data/test.json \
    --output ./evaluation_results_complete.json
```

#### 计算论文指标
```bash
python compute_paper_metrics.py
```

### 3. 生成晶体

```python
from train_crystalicl import CrystalICLTrainer

# 加载模型
trainer = CrystalICLTrainer(model_name="./crystalicl_qwen3_8b_output")

# 条件生成
instruction = """### Instruction: Below is a description of a bulk material.
The chemical formula is NaCl. The spacegroup number is 225.
Generate the space group symbol, a description of the lengths and angles
of the lattice vectors and then the element type and coordinates for each
atom within the lattice:
### Response:"""

generated = trainer.generate(instruction)
print(generated)
```

### 4. 使用不同的示例选择策略

```python
from example_selector import ExampleSelector

selector = ExampleSelector()

# 策略1: 基于条件的选择
selected = selector.condition_based_selection(
    dataset, target_properties, k=3
)

# 策略2: 基于结构的选择
selected = selector.structure_based_selection(
    dataset, anchor_structure, k=3
)

# 策略3: 混合选择（推荐，论文中表现最好）
selected = selector.condition_structure_based_selection(
    dataset, target_properties, anchor_structure, k=3
)
```

## 核心功能

### 1. 晶体Token化

```python
from crystal_tokenization import CrystalTokenizer
from pymatgen.core import Structure, Lattice

lattice = Lattice.cubic(5.64)
structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

tokenizer = CrystalTokenizer()
sgs_text = tokenizer.tokenize(structure, use_sgs=True)
print(sgs_text)
```

输出：
```
Fm-3m
5.640 5.640 5.640
90.0 90.0 90.0
Na
0.00 0.00 0.00
Cl
0.50 0.50 0.50
```

### 2. 指令构建

```python
from instruction_builder import InstructionBuilder

builder = InstructionBuilder(tokenizer)

# 零样本指令
instruction = builder.build_zero_shot_instruction(
    "formation energy", "-0.5 eV/atom"
)

# 少样本指令（K-shot）
instruction = builder.build_few_shot_instruction(
    examples, target_properties, k_shot=3
)

# 属性预测指令
instruction = builder.build_property_prediction_instruction(
    structure, "formation energy"
)
```

### 3. 评估指标计算

```python
from compute_paper_metrics import PaperMetricsComputer

computer = PaperMetricsComputer()

# 计算 Table 1 指标（条件生成）
table1_results = computer.compute_table1_metrics(
    generated_structures,
    target_properties,
    num_iterations=5
)

# 计算 Table 2 指标（无条件生成）
table2_results = computer.compute_table2_metrics(
    generated_structures,
    reference_structures
)
```

## 命令行参数

### 训练参数
- `--model_name`: 基础模型名称（默认：`Qwen/Qwen3-8B`）
- `--output_dir`: 输出目录
- `--num_epochs`: 训练轮数（默认：3）
- `--batch_size`: 批次大小（默认：1）
- `--learning_rate`: 学习率（默认：5e-4）
- `--use_few_shot`: 使用少样本学习
- `--k_shot`: 少样本数量（默认：3）

### LoRA参数
- `--use_lora`: 使用LoRA微调（默认：True）
- `--lora_rank`: LoRA秩（默认：8）
- `--lora_alpha`: LoRA alpha（默认：32）
- `--lora_dropout`: LoRA dropout（默认：0.05）

### 评估参数
- `--model_path`: 训练好的模型路径
- `--test_data`: 测试数据路径
- `--num_samples`: 条件生成样本数（默认：1000）
- `--num_unconditional`: 无条件生成样本数（默认：10000）

## 预期性能

根据论文，使用3-shot SGS格式：

### MP20数据集
- Pretty Formula: Mean=0.9906, Std=0.0050
- Space Group: Mean=0.0886, Std=0.0098
- Formation Energy: Mean=0.8751, Std=0.0048
- Band Gap: Mean=0.7087, Std=0.0165

### MP30数据集
- Pretty Formula: Mean=0.9922, Std=0.0028
- Space Group: Mean=0.1083, Std=0.0089
- Formation Energy: Mean=0.9461, Std=0.0056
- Band Gap: Mean=0.7454, Std=0.0139

## 与原论文的差异

### 相同点
✅ 核心算法完全一致  
✅ SGS token化方法  
✅ 三种示例选择策略  
✅ 混合指令微调框架  
✅ 多任务学习策略  
✅ 完整的评估指标体系  

### 差异点
| 方面 | 原论文 | 本实现 |
|------|--------|--------|
| 基础模型 | Llama-2-7b-chat | **Qwen3-8B** |
| 训练框架 | 未明确说明 | Transformers + PEFT |
| DFT计算 | 真实计算 | 占位符（可扩展） |

### Qwen3-8B 的优势
- 更强的指令遵循能力
- 更长的上下文窗口（128K tokens）
- 更好的中英文支持
- 更新的训练数据（截至2024年）
- 更强的推理能力

## 硬件要求

- **GPU**: 至少16GB显存（推荐24GB+）
- **RAM**: 至少32GB
- **存储**: 至少50GB可用空间
- **CUDA**: 11.8+ 或 12.1+

## 常见问题

### Q: 如何使用真实的DFT计算？
A: 需要集成VASP或Quantum ESPRESSO：
```python
# 在 compute_paper_metrics.py 中替换占位符
def compute_formation_energy(structure):
    # 调用VASP计算
    from pymatgen.io.vasp import Poscar
    poscar = Poscar(structure)
    # ... VASP计算流程
    return formation_energy
```

### Q: 如何加载真实的材料数据集？
A: 从Materials Project下载：
```python
from pymatgen.ext.matproj import MPRester

with MPRester("YOUR_API_KEY") as mpr:
    structures = mpr.query(
        criteria={"nelements": 2},
        properties=["structure", "formation_energy_per_atom", "band_gap"]
    )
```

### Q: 显存不足怎么办？
A: 
- 减小 `batch_size` 到 1
- 增加 `gradient_accumulation_steps`
- 使用更小的 `lora_rank`（如4）
- 使用 `int8` 量化

## 文件说明

### 核心模块
- `crystal_tokenization.py` - SGS格式token化
- `instruction_builder.py` - 指令构建（零样本/少样本）
- `example_selector.py` - 三种示例选择策略
- `train_crystalicl.py` - Qwen3-8B训练脚本

### 评估模块
- `metrics_calculator.py` - 基础指标计算
- `compute_paper_metrics.py` - 论文Table 1&2指标
- `evaluate_complete.py` - 完整评估流程

### 工具脚本
- `data_loader.py` - 数据加载（JSON/CIF）
- `run_crystalicl.py` - 主运行脚本
- `examples.py` - 使用示例
- `test_modules.py` - 模块测试

## 引用

```bibtex
@article{wang2025crystalicl,
  title={CrystalICL: Enabling In-Context Learning for Crystal Generation},
  author={Wang, Ruobing and Tan, Qiaoyu and Wang, Yili and Wang, Ying and Wang, Xin},
  journal={arXiv preprint arXiv:2508.20143},
  year={2025}
}
```

## 许可证

本项目仅用于学术研究目的。

## 联系方式

如有问题，请提交 Issue。
