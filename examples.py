"""
Example usage of CrystalICL
展示如何使用CrystalICL进行晶体生成
"""

from train_crystalicl import CrystalICLTrainer
from crystal_tokenization import CrystalTokenizer
from instruction_builder import InstructionBuilder
from pymatgen.core import Structure


def example_1_generate_with_properties():
    """示例1: 根据属性生成晶体"""
    print("="*80)
    print("Example 1: Generate Crystal with Specific Properties")
    print("="*80)

    # 加载训练好的模型
    trainer = CrystalICLTrainer(
        model_name="./crystalicl_qwen_output",  # 使用训练好的模型
        use_lora=True
    )

    # 构建指令
    instruction = """### Instruction: Below is a description of a bulk material.
The chemical formula is NaCl. The spacegroup number is 225.
The formation energy per atom is -0.5. The band gap is 5.0.
Generate the space group symbol, a description of the lengths and angles
of the lattice vectors and then the element type and coordinates for each
atom within the lattice:
### Response:"""

    # 生成晶体
    print("\nGenerating crystal structure...")
    generated_text = trainer.generate(
        instruction,
        max_new_tokens=512,
        temperature=0.9,
        top_p=0.9
    )

    print("\nGenerated Structure:")
    print(generated_text)


def example_2_few_shot_generation():
    """示例2: 少样本生成"""
    print("\n" + "="*80)
    print("Example 2: Few-shot Crystal Generation")
    print("="*80)

    from pymatgen.core import Lattice
    from example_selector import ExampleSelector

    # 准备示例数据
    examples = []
    for i in range(3):
        lattice = Lattice.cubic(5.64 + i * 0.1)
        structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])
        examples.append({
            'structure': structure,
            'properties': {
                'chemical_formula': 'NaCl',
                'spacegroup': 225,
                'formation_energy': -0.5 - i * 0.05,
                'band_gap': 5.0 + i * 0.1
            }
        })

    # 构建少样本指令
    tokenizer = CrystalTokenizer()
    builder = InstructionBuilder(tokenizer)

    target_properties = {
        'chemical_formula': 'NaCl',
        'spacegroup': 225,
        'formation_energy': -0.55,
        'band_gap': 5.15
    }

    instruction = builder.build_few_shot_instruction(
        examples,
        target_properties,
        k_shot=3
    )

    print("\nFew-shot Instruction (first 500 chars):")
    print(instruction[:500] + "...\n")

    # 加载模型并生成
    trainer = CrystalICLTrainer(
        model_name="./crystalicl_qwen_output",
        use_lora=True
    )

    print("Generating with few-shot examples...")
    generated_text = trainer.generate(instruction)

    print("\nGenerated Structure:")
    print(generated_text)


def example_3_property_prediction():
    """示例3: 属性预测"""
    print("\n" + "="*80)
    print("Example 3: Property Prediction")
    print("="*80)

    from pymatgen.core import Lattice

    # 创建晶体结构
    lattice = Lattice.cubic(5.64)
    structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # 构建属性预测指令
    tokenizer = CrystalTokenizer()
    builder = InstructionBuilder(tokenizer)

    instruction = builder.build_property_prediction_instruction(
        structure,
        "formation energy"
    )

    print("\nProperty Prediction Instruction:")
    print(instruction[:300] + "...\n")

    # 加载模型并预测
    trainer = CrystalICLTrainer(
        model_name="./crystalicl_qwen_output",
        use_lora=True
    )

    print("Predicting formation energy...")
    predicted_value = trainer.generate(instruction, max_new_tokens=50)

    print("\nPredicted Formation Energy:")
    print(predicted_value)


def example_4_batch_generation():
    """示例4: 批量生成"""
    print("\n" + "="*80)
    print("Example 4: Batch Crystal Generation")
    print("="*80)

    trainer = CrystalICLTrainer(
        model_name="./crystalicl_qwen_output",
        use_lora=True
    )

    # 定义多个目标属性
    targets = [
        {
            'chemical_formula': 'NaCl',
            'spacegroup': 225,
            'formation_energy': -0.5
        },
        {
            'chemical_formula': 'MgO',
            'spacegroup': 225,
            'formation_energy': -0.6
        },
        {
            'chemical_formula': 'CaF2',
            'spacegroup': 225,
            'formation_energy': -0.7
        }
    ]

    tokenizer = CrystalTokenizer()
    builder = InstructionBuilder(tokenizer)

    print(f"\nGenerating {len(targets)} crystal structures...\n")

    for i, target in enumerate(targets, 1):
        print(f"Target {i}: {target['chemical_formula']}")

        instruction = builder.build_conditional_generation_instruction(
            target,
            use_few_shot=False
        )

        generated_text = trainer.generate(instruction, max_new_tokens=256)

        print(f"Generated (first 200 chars):")
        print(generated_text[:200] + "...\n")


def example_5_structure_analysis():
    """示例5: 结构分析"""
    print("\n" + "="*80)
    print("Example 5: Analyze Generated Structure")
    print("="*80)

    from pymatgen.core import Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    # 创建示例结构
    lattice = Lattice.cubic(5.64)
    structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

    print("\nOriginal Structure:")
    print(f"Formula: {structure.composition.reduced_formula}")
    print(f"Lattice: a={lattice.a:.3f}, b={lattice.b:.3f}, c={lattice.c:.3f}")
    print(f"Density: {structure.density:.3f} g/cm³")
    print(f"Volume: {structure.volume:.3f} Ų")

    # 空间群分析
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    print(f"\nSpace Group: {sga.get_space_group_symbol()}")
    print(f"Space Group Number: {sga.get_space_group_number()}")
    print(f"Crystal System: {sga.get_crystal_system()}")
    print(f"Point Group: {sga.get_point_group_symbol()}")

    # Token化
    tokenizer = CrystalTokenizer()
    sgs_text = tokenizer.tokenize(structure, use_sgs=True)

    print("\nSGS Tokenization:")
    print(sgs_text)


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("CrystalICL Usage Examples")
    print("="*80)

    try:
        # 注意：这些示例需要先训练模型
        print("\nNote: These examples require a trained model.")
        print("Please run 'python run_crystalicl.py --do_train' first.\n")

        # 运行示例5（不需要训练好的模型）
        example_5_structure_analysis()

        # 如果模型已训练，可以运行其他示例
        import os
        if os.path.exists("./crystalicl_qwen_output"):
            print("\nTrained model found! Running generation examples...\n")
            example_1_generate_with_properties()
            example_2_few_shot_generation()
            example_3_property_prediction()
            example_4_batch_generation()
        else:
            print("\nTrained model not found. Skipping generation examples.")
            print("Run 'python run_crystalicl.py --do_train' to train the model first.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
