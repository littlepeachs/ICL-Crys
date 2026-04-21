# CrystalICL 完整使用指南 - 基于 Qwen3-8B

## 目录
1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [完整评估](#完整评估)
5. [计算论文指标](#计算论文指标)
6. [结果解读](#结果解读)

---

## 环境准备

### 1. 创建环境并安装依赖

```bash
# 创建conda环境
conda create -n crystalicl python=3.10
conda activate crystalicl

# 安装依赖
pip install -r requirements.txt

# 安装PyTorch（根据CUDA版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 测试安装

```bash
python test_modules.py
```

---

## 数据准备

### 选项1: 使用示例数据（快速测试）

```bash
python data_loader.py
```

这将创建100个示例晶体结构用于测试。

### 选项2: 使用真实数据集

#### 从Materials Project下载

```python
from pymatgen.ext.matproj import MPRester
from data_loader import CrystalDataLoader

# 下载数据
with MPRester("YOUR_API_KEY") as mpr:
    data = mpr.query(
        criteria={"nelements": {"$lte": 3}},
        properties=["structure", "formation_energy_per_atom", "band_gap", "spacegroup"]
    )

# 转换格式
loader = CrystalDataLoader()
processed_data = []
for entry in data:
    processed_data.append({
        'structure': entry['structure'],
        'properties': {
            'chemical_formula': entry['structure'].composition.reduced_formula,
            'spacegroup': entry['spacegroup']['number'],
            'formation_energy': entry['formation_energy_per_atom'],
            'band_gap': entry['band_gap']
        }
    })

# 保存
loader.save_to_json(processed_data, './data/mp_data.json')
```

#### 从CIF文件加载

```bash
python run_crystalicl.py \
    --data_path ./cif_directory \
    --data_format cif \
    --properties_file ./properties.json
```

---

## 模型训练

### 1. 基础训练（使用示例数据）

```bash
python run_crystalicl.py \
    --model_name Qwen/Qwen3-8B \
    --use_sample_data \
    --num_samples 100 \
    --do_train \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --use_few_shot \
    --k_shot 3 \
    --output_dir ./crystalicl_qwen3_8b_output
```

### 2. 使用真实数据训练

```bash
python run_crystalicl.py \
    --model_name Qwen/Qwen3-8B \
    --data_path ./data/mp_data.json \
    --data_format json \
    --do_train \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-4 \
    --use_few_shot \
    --k_shot 3 \
    --output_dir ./crystalicl_qwen3_8b_output
```

### 3. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | Qwen/Qwen3-8B | 基础模型 |
| `--num_epochs` | 3 | 训练轮数 |
| `--batch_size` | 1 | 批次大小 |
| `--learning_rate` | 5e-4 | 学习率 |
| `--use_lora` | True | 使用LoRA微调 |
| `--lora_rank` | 8 | LoRA秩 |
| `--k_shot` | 3 | 少样本数量 |

---

## 完整评估

### 1. 运行完整评估（计算所有论文指标）

```bash
python evaluate_complete.py \
    --model_path ./crystalicl_qwen3_8b_output \
    --test_data ./data/test.json \
    --output ./evaluation_results_complete.json \
    --num_samples 1000 \
    --num_unconditional 10000
```

这将计算：
- **Table 1**: 条件生成指标（1000个样本）
- **Table 2**: 无条件生成指标（10000个样本）

### 2. 评估流程说明

#### Table 1: 条件生成评估

对每个测试样本：
1. 根据目标属性构建指令
2. 生成晶体结构
3. 检查生成结构是否匹配目标属性
4. 计算成功率的均值和标准差

评估指标：
- **Pretty Formula**: 化学式匹配
- **Space Group**: 空间群匹配
- **Formation Energy**: 形成能匹配（需DFT）
- **Band Gap**: 带隙匹配（需DFT）

#### Table 2: 无条件生成评估

生成10000个样本，计算：
1. **Validity Check**: 结构有效性
   - Structural Validity（原子间距）
   - Compositional Validity（电荷中性）
   
2. **Coverage**: 覆盖率
   - Recall（覆盖了多少参考结构）
   - Precision（有效结构比例）
   
3. **Property Distribution**: 属性分布
   - Wasserstein距离（密度、原子数等）

---

## 计算论文指标

### 方法1: 使用完整评估脚本

```bash
python evaluate_complete.py \
    --model_path ./crystalicl_qwen3_8b_output \
    --test_data ./data/test.json
```

### 方法2: 使用指标计算脚本

```python
from compute_paper_metrics import PaperMetricsComputer
from data_loader import CrystalDataLoader

# 加载数据
loader = CrystalDataLoader()
test_data = loader.load_from_json('./data/test.json')

# 生成结构（这里使用测试数据作为示例）
generated_structures = [s['structure'] for s in test_data]
target_properties = [s['properties'] for s in test_data]
reference_structures = [s['structure'] for s in test_data]

# 计算指标
computer = PaperMetricsComputer()

# Table 1
table1_results = computer.compute_table1_metrics(
    generated_structures,
    target_properties,
    num_iterations=5
)

# Table 2
table2_results = computer.compute_table2_metrics(
    generated_structures,
    reference_structures
)

print("Table 1 Results:")
print(table1_results)

print("\nTable 2 Results:")
print(table2_results)
```

### 方法3: 分步计算

```python
from metrics_calculator import CrystalMetricsCalculator

calculator = CrystalMetricsCalculator()

# 1. 计算有效性
validity = calculator.compute_validity_metrics(generated_structures)
print("Validity:", validity)

# 2. 计算覆盖率
coverage = calculator.compute_coverage_metrics(
    generated_structures,
    reference_structures
)
print("Coverage:", coverage)

# 3. 计算属性分布
distribution = calculator.compute_property_distribution_metrics(
    generated_structures,
    reference_structures
)
print("Distribution:", distribution)

# 4. 计算条件成功率
for prop in ['chemical_formula', 'spacegroup']:
    mean, std = calculator.compute_conditional_success_rate(
        generated_structures,
        target_properties,
        prop
    )
    print(f"{prop}: Mean={mean:.4f}, Std={std:.4f}")
```

---

## 结果解读

### Table 1 结果示例

```
Table 1: Conditional Sample Performance
================================================================================

Property                  Mean            Std            
--------------------------------------------------------------------------------
Pretty Formula            0.9906          0.0050
Space Group               0.0886          0.0098
Formation Energy          0.8751          0.0048
Band Gap                  0.7087          0.0165
```

**解读**：
- **Pretty Formula (99.06%)**: 化学式匹配率很高，说明模型能准确生成目标化学组成
- **Space Group (8.86%)**: 空间群匹配率较低，这是正常的，因为同一化学式可能有多个空间群
- **Formation Energy (87.51%)**: 形成能匹配率高（需要DFT计算验证）
- **Band Gap (70.87%)**: 带隙匹配率较好（需要DFT计算验证）

### Table 2 结果示例

```
Table 2: Unconditional Sample Performance
================================================================================

Validity Check:
----------------------------------------
  structural_validity            0.9420
  compositional_validity         0.9850
  total_validity                 0.9280

Coverage:
----------------------------------------
  recall                         0.7234
  precision                      0.9156

Property Distribution (Wasserstein Distance):
----------------------------------------
  density                        0.1234
  num_atoms                      2.3456
  volume                         15.678
  num_elements                   0.5678
```

**解读**：
- **Validity**: 94.2%的结构有效，说明生成质量高
- **Coverage**: 覆盖了72.34%的参考结构，精确率91.56%
- **Distribution**: Wasserstein距离越小越好，说明生成分布接近真实分布

---

## 高级用法

### 1. 使用不同的示例选择策略

```python
from example_selector import ExampleSelector

selector = ExampleSelector()

# 条件选择（C）
examples_c = selector.condition_based_selection(dataset, target_props, k=3)

# 结构选择（F）
examples_f = selector.structure_based_selection(dataset, anchor, k=3)

# 混合选择（CF）- 推荐
examples_cf = selector.condition_structure_based_selection(
    dataset, target_props, anchor, k=3
)
```

### 2. 批量生成和评估

```python
from train_crystalicl import CrystalICLTrainer
from tqdm import tqdm

trainer = CrystalICLTrainer(model_name="./crystalicl_qwen3_8b_output")

# 批量生成
generated_structures = []
for target in tqdm(target_list):
    instruction = build_instruction(target)
    generated = trainer.generate(instruction)
    structure = parse_structure(generated)
    generated_structures.append(structure)

# 批量评估
results = evaluate_all_metrics(generated_structures, references)
```

### 3. 集成DFT计算

```python
def compute_formation_energy_dft(structure):
    """使用VASP计算formation energy"""
    from pymatgen.io.vasp import Poscar, Incar, Kpoints
    from pymatgen.io.vasp.sets import MPRelaxSet
    
    # 创建VASP输入
    vasp_input = MPRelaxSet(structure)
    vasp_input.write_input("./vasp_calc")
    
    # 运行VASP（需要配置VASP环境）
    # os.system("cd vasp_calc && vasp_std")
    
    # 读取结果
    # from pymatgen.io.vasp import Vasprun
    # vasprun = Vasprun("./vasp_calc/vasprun.xml")
    # formation_energy = vasprun.final_energy / len(structure)
    
    return formation_energy

# 在评估中使用
def check_formation_energy_match(gen_struct, target_fe, tolerance=0.5):
    gen_fe = compute_formation_energy_dft(gen_struct)
    return abs(gen_fe - target_fe) < tolerance
```

---

## 性能优化建议

### 1. 显存优化

```bash
# 减小batch size
--batch_size 1

# 增加梯度累积
--gradient_accumulation_steps 16

# 使用更小的LoRA rank
--lora_rank 4

# 使用int8量化
# 在train_crystalicl.py中添加：
# load_in_8bit=True
```

### 2. 速度优化

```bash
# 使用更少的训练样本
--num_samples 50

# 减少评估样本
--num_samples 100 --num_unconditional 1000

# 使用多GPU
# 在TrainingArguments中添加：
# ddp_find_unused_parameters=False
```

### 3. 质量优化

```bash
# 增加训练轮数
--num_epochs 10

# 使用更多的few-shot示例
--k_shot 5

# 使用混合示例选择策略
# 在代码中设置：
# example_selector.condition_structure_based_selection()
```

---

## 故障排除

### 问题1: CUDA out of memory

**解决方案**：
```bash
# 减小batch size
--batch_size 1

# 增加梯度累积
--gradient_accumulation_steps 16

# 使用CPU（慢）
export CUDA_VISIBLE_DEVICES=""
```

### 问题2: 生成的结构无效

**检查**：
1. 模型是否训练充分
2. 指令格式是否正确
3. 解析函数是否正确

**调试**：
```python
# 打印生成的原始文本
print("Generated text:", generated_text)

# 检查解析结果
structure = parse_generated_structure(generated_text)
if structure is None:
    print("Parsing failed!")
```

### 问题3: 评估指标异常

**检查**：
1. 测试数据是否正确加载
2. 生成的结构是否有效
3. 参考结构是否正确

**调试**：
```python
# 检查数据
print(f"Generated: {len(generated_structures)}")
print(f"Valid: {sum(1 for s in generated_structures if s is not None)}")

# 检查单个样本
sample = generated_structures[0]
if sample:
    print(f"Formula: {sample.composition.reduced_formula}")
    print(f"Density: {sample.density}")
```

---

## 总结

本指南涵盖了从环境准备到完整评估的所有步骤。关键要点：

1. ✅ 使用 **Qwen3-8B** 作为基础模型
2. ✅ 实现了论文中 **所有评估指标**
3. ✅ 支持 **零样本和少样本** 学习
4. ✅ 提供 **三种示例选择策略**
5. ✅ 完整的 **Table 1 和 Table 2** 指标计算

如有问题，请查看代码注释或提交Issue。
