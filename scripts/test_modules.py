"""
Test script to verify all modules
测试所有模块是否正常工作
"""

import sys
import traceback


def test_crystal_tokenization():
    """测试晶体token化模块"""
    print("\n" + "="*80)
    print("Testing Crystal Tokenization Module")
    print("="*80)

    try:
        from crystal_tokenization import CrystalTokenizer
        from pymatgen.core import Structure, Lattice

        # 创建测试结构
        lattice = Lattice.cubic(5.64)
        structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

        tokenizer = CrystalTokenizer()

        # 测试SGS格式
        sgs_text = tokenizer.tokenize(structure, use_sgs=True)
        print("SGS Format:")
        print(sgs_text)

        # 测试XYZ格式
        xyz_text = tokenizer.tokenize(structure, use_sgs=False)
        print("\nXYZ Format:")
        print(xyz_text)

        print("\n✓ Crystal Tokenization Module: PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Crystal Tokenization Module: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_instruction_builder():
    """测试指令构建模块"""
    print("\n" + "="*80)
    print("Testing Instruction Builder Module")
    print("="*80)

    try:
        from instruction_builder import InstructionBuilder
        from crystal_tokenization import CrystalTokenizer
        from pymatgen.core import Structure, Lattice

        tokenizer = CrystalTokenizer()
        builder = InstructionBuilder(tokenizer)

        # 测试零样本指令
        zero_shot = builder.build_zero_shot_instruction(
            "formation energy",
            "-0.5 eV/atom"
        )
        print("Zero-shot Instruction (first 200 chars):")
        print(zero_shot[:200] + "...")

        # 测试少样本指令
        lattice = Lattice.cubic(5.64)
        structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

        examples = [{
            'structure': structure,
            'properties': {
                'chemical_formula': 'NaCl',
                'spacegroup': 225,
                'formation_energy': -0.5
            }
        }] * 3

        target_props = {
            'chemical_formula': 'MgO',
            'spacegroup': 225
        }

        few_shot = builder.build_few_shot_instruction(examples, target_props, k_shot=3)
        print("\nFew-shot Instruction (first 200 chars):")
        print(few_shot[:200] + "...")

        print("\n✓ Instruction Builder Module: PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Instruction Builder Module: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_example_selector():
    """测试示例选择模块"""
    print("\n" + "="*80)
    print("Testing Example Selector Module")
    print("="*80)

    try:
        from example_selector import ExampleSelector
        from pymatgen.core import Structure, Lattice

        # 创建测试数据集
        dataset = []
        for i in range(10):
            lattice = Lattice.cubic(5.64 + i * 0.1)
            structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])
            dataset.append({
                'structure': structure,
                'properties': {
                    'chemical_formula': 'NaCl',
                    'spacegroup': 225,
                    'formation_energy': -0.5 - i * 0.1
                }
            })

        selector = ExampleSelector()

        # 测试条件选择
        target_props = {'chemical_formula': 'NaCl', 'spacegroup': 225}
        selected = selector.condition_based_selection(dataset, target_props, k=3)
        print(f"Condition-based selection: {len(selected)} samples selected")

        # 测试结构选择
        anchor = dataset[0]['structure']
        selected = selector.structure_based_selection(dataset, anchor, k=3)
        print(f"Structure-based selection: {len(selected)} samples selected")

        # 测试混合选择
        selected = selector.condition_structure_based_selection(
            dataset, target_props, anchor, k=3
        )
        print(f"Hybrid selection: {len(selected)} samples selected")

        print("\n✓ Example Selector Module: PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Example Selector Module: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_data_loader():
    """测试数据加载模块"""
    print("\n" + "="*80)
    print("Testing Data Loader Module")
    print("="*80)

    try:
        from data_loader import CrystalDataLoader
        import os

        loader = CrystalDataLoader(data_dir="./test_data")

        # 创建示例数据
        data = loader.create_sample_dataset(num_samples=20)
        print(f"Created {len(data)} sample structures")

        # 测试数据划分
        splits = loader.split_dataset(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        # 测试保存和加载
        os.makedirs("./test_data", exist_ok=True)
        loader.save_to_json(splits['train'], "./test_data/test_train.json")
        loaded_data = loader.load_from_json("./test_data/test_train.json")
        print(f"Saved and loaded {len(loaded_data)} samples")

        # 清理测试文件
        import shutil
        if os.path.exists("./test_data"):
            shutil.rmtree("./test_data")

        print("\n✓ Data Loader Module: PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Data Loader Module: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_dependencies():
    """测试依赖包"""
    print("\n" + "="*80)
    print("Testing Dependencies")
    print("="*80)

    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'pymatgen': 'Pymatgen',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'tqdm': 'tqdm'
    }

    all_passed = True

    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}: installed")
        except ImportError:
            print(f"✗ {name}: NOT installed")
            all_passed = False

    if all_passed:
        print("\n✓ All Dependencies: PASSED")
    else:
        print("\n✗ Some Dependencies: MISSING")

    return all_passed


def main():
    """运行所有测试"""
    print("="*80)
    print("CrystalICL Module Tests")
    print("="*80)

    results = {}

    # 测试依赖
    results['dependencies'] = test_dependencies()

    # 测试各个模块
    results['tokenization'] = test_crystal_tokenization()
    results['instruction'] = test_instruction_builder()
    results['selector'] = test_example_selector()
    results['data_loader'] = test_data_loader()

    # 总结
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    for module, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{module.capitalize()}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("All tests PASSED! ✓")
        print("You can now run the training script.")
    else:
        print("Some tests FAILED! ✗")
        print("Please check the errors above and install missing dependencies.")
    print("="*80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
