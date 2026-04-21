# CrystalICL: 基于Qwen3-8B的晶体生成模型

这是论文 "CrystalICL: Enabling In-Context Learning for Crystal Generation" 的复现代码，使用 Qwen3-8B (Qwen2.5-7B-Instruct) 作为基础模型。

## 论文概述

CrystalICL 是一个专门为晶体生成设计的少样本学习模型，主要创新点包括：

1. **Space-group based Crystal Tokenization (SGS)** - 基于空间群的晶体token化方法，简化了晶体对称性建模
2. **Condition-Structure Aware Hybrid Crystal Instruction Tuning** - 条件-结构感知的混合指令微调框架
3. **Multi-Task Crystal Instruction Tuning** - 多任务学习策略，同时训练晶体生成和属性预测

## 项目结构

```
ICL-Crys/
├── crystal_tokenization.py      # 晶体token化模块（SGS格式）
├── instruction_builder.py       # 指令构建模块
├── example_selector.py          # 示例选择策略（3种）
├── data_loader.py              # 数据加载器
├── train_crystalicl.py         # 训练脚本
├── evaluate.py                 # 评估脚本
├── run_crystalicl.py          # 主运行脚本
├── requirements.txt            # 依赖包
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

# CPU only
pip install torch torchvision torchaudio
```

## 快速开始

### 1. 使用示例数据测试

```bash
# 准备示例数据、训练和评估
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

### 2. 使用自己的数据

#### 从JSON文件加载

```bash
python run_crystalicl.py \
    --data_path ./your_data.json \
    --data_format json \
    --do_train \
    --do_eval
```

JSON格式示例：
```json
[
    {
        "structure": {
            "lattice": {...},
            "sites": [...]
        },
        "properties": {
            "formation_energy": -0.5,
            "band_gap": 5.0,
            "spacegroup": 225,
            "chemical_formula": "NaCl"
        }
    }
]
```

#### 从CIF文件加载

```bash
python run_crystalicl.py \
    --data_path ./cif_directory \
    --data_format cif \
    --properties_file ./properties.json \
    --do_train \
    --do_eval
```

## 核心功能

### 1. 晶体Token化

```python
from crystal_tokenization import CrystalTokenizer
from pymatgen.core import Structure, Lattice

# 创建晶体结构
lattice = Lattice.cubic(5.64)
structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

# Token化
tokenizer = CrystalTokenizer()
sgs_text = tokenizer.tokenize(structure, use_sgs=True)
print(sgs_text)
```

输出示例：
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
    "formation energy",
    "-0.5 eV/atom"
)

# 少样本指令
instruction = builder.build_few_shot_instruction(
    examples=examples,
    target_property=target_properties,
    k_shot=3
)
```

### 3. 示例选择策略

```python
from example_selector import ExampleSelector

selector = ExampleSelector()

# 基于条件的选择
selected = selector.condition_based_selection(
    dataset, target_properties, k=3
)

# 基于结构的选择
selected = selector.structure_based_selection(
    dataset, anchor_structure, k=3
)

# 混合选择（推荐）
selected = selector.condition_structure_based_selection(
    dataset, target_properties, anchor_structure, k=3
)
```

### 4. 训练模型

```python
from train_crystalicl import CrystalICLTrainer

trainer = CrystalICLTrainer(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    output_dir="./output",
    use_lora=True,
    lora_rank=8
)

trainer.train(
    train_data=train_data,
    eval_data=val_data,
    num_epochs=3,
    batch_size=1,
    use_few_shot=True,
    k_shot=3
)
```

### 5. 生成晶体

```python
# 条件生成
instruction = """### Instruction: Below is a description of a bulk material. 
The chemical formula is NaCl. The spacegroup number is 225. 
Generate the space group symbol, a description of the lengths and angles 
of the lattice vectors and then the element type and coordinates for each 
atom within the lattice:
### Response:"""

generated_text = trainer.generate(instruction)
print(generated_text)
```

### 6. 评估模型

```python
from evaluate import CrystalEvaluator

evaluator = CrystalEvaluator(model_path="./output")

results = evaluator.evaluate_model(
    test_data=test_data,
    num_samples=100
)

print(results)
```

## 命令行参数

### 数据参数
- `--data_dir`: 数据目录（默认：`./data`）
- `--data_path`: 数据文件或目录路径
- `--data_format`: 数据格式（`json` 或 `cif`）
- `--use_sample_data`: 使用示例数据
- `--num_samples`: 示例数据样本数量

### 模型参数
- `--model_name`: 基础模型名称（默认：`Qwen/Qwen2.5-7B-Instruct`）
- `--output_dir`: 输出目录
- `--use_lora`: 使用LoRA微调
- `--lora_rank`: LoRA秩（默认：8）
- `--lora_alpha`: LoRA alpha（默认：32）

### 训练参数
- `--num_epochs`: 训练轮数（默认：3）
- `--batch_size`: 批次大小（默认：1）
- `--learning_rate`: 学习率（默认：5e-4）
- `--use_few_shot`: 使用少样本学习
- `--k_shot`: 少样本数量（默认：3）

### 运行模式
- `--do_train`: 执行训练
- `--do_eval`: 执行评估
- `--eval_samples`: 评估样本数量

## 评估指标

模型评估包括以下指标：

1. **成功率 (Success Rate)**
   - Pretty Formula Match
   - Space Group Match
   - Formation Energy Match
   - Band Gap Match

2. **有效性指标 (Validity Metrics)**
   - Structural Validity（结构有效性）
   - Compositional Validity（组成有效性）

3. **属性分布 (Property Distribution)**
   - Wasserstein Distance（密度、原子数等）

## 与原论文的差异

1. **模型选择**: 使用 Qwen2.5-7B-Instruct 替代 Llama-2-7b-chat
2. **简化实现**: 某些复杂的物理化学计算（如DFT）使用占位符
3. **数据集**: 提供示例数据生成器，可以替换为真实的MP20/MP30/P5/C24数据集

## 性能优化建议

1. **使用混合精度训练**: 已启用 `bf16=True`
2. **梯度累积**: 设置 `gradient_accumulation_steps=8`
3. **LoRA微调**: 减少显存占用，加快训练速度
4. **批处理**: 根据GPU显存调整 `batch_size`

## 常见问题

### Q: 显存不足怎么办？
A: 
- 减小 `batch_size`
- 增加 `gradient_accumulation_steps`
- 使用更小的 `lora_rank`
- 使用 `int8` 量化

### Q: 如何使用真实的材料数据集？
A: 
1. 从 Materials Project 下载数据
2. 转换为JSON格式
3. 使用 `--data_path` 参数加载

### Q: 如何改进生成质量？
A:
- 增加训练数据量
- 使用更大的基础模型
- 调整 `k_shot` 参数
- 使用混合示例选择策略

## 引用

如果使用本代码，请引用原论文：

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

如有问题，请提交 Issue 或联系作者。
