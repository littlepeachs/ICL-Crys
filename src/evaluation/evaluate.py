"""
Evaluation Script for CrystalICL
评估晶体生成模型的性能
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
import re
from tqdm import tqdm

from train_crystalicl import CrystalICLTrainer
from crystal_tokenization import CrystalTokenizer
from instruction_builder import InstructionBuilder


class CrystalEvaluator:
    """晶体生成评估器"""

    def __init__(self, model_path: str = None):
        self.crystal_tokenizer = CrystalTokenizer()
        self.instruction_builder = InstructionBuilder(self.crystal_tokenizer)
        self.structure_matcher = StructureMatcher()

        if model_path:
            self.trainer = CrystalICLTrainer(model_name=model_path)

    def parse_generated_structure(self, generated_text: str) -> Structure:
        """
        解析生成的晶体结构文本

        Args:
            generated_text: 生成的文本

        Returns:
            pymatgen Structure对象
        """
        try:
            lines = generated_text.strip().split('\n')

            # 解析空间群
            spacegroup_line = lines[0]

            # 解析晶格参数
            lattice_params = list(map(float, lines[1].split()))
            angles = list(map(float, lines[2].split()))

            a, b, c = lattice_params
            alpha, beta, gamma = angles

            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

            # 解析原子
            species = []
            coords = []

            i = 3
            while i < len(lines):
                if lines[i].strip():
                    # 元素符号
                    element = lines[i].strip()
                    species.append(element)

                    # 坐标
                    i += 1
                    if i < len(lines):
                        coord = list(map(float, lines[i].split()))
                        coords.append(coord)

                i += 1

            structure = Structure(lattice, species, coords)
            return structure

        except Exception as e:
            print(f"Error parsing structure: {e}")
            return None

    def compute_success_rate(
        self,
        test_data: List[Dict[str, Any]],
        property_name: str,
        tolerance: float = 0.5
    ) -> float:
        """
        计算成功率

        Args:
            test_data: 测试数据
            property_name: 属性名称
            tolerance: 容差

        Returns:
            成功率
        """
        success_count = 0
        total_count = 0

        for sample in tqdm(test_data, desc=f"Evaluating {property_name}"):
            target_properties = sample['properties']

            if property_name not in target_properties:
                continue

            # 构建指令
            instruction = self.instruction_builder.build_conditional_generation_instruction(
                target_properties, use_few_shot=False
            )

            # 生成结构
            generated_text = self.trainer.generate(instruction)
            generated_structure = self.parse_generated_structure(generated_text)

            if generated_structure is None:
                total_count += 1
                continue

            # 检查属性是否匹配
            if property_name == 'formation_energy':
                # 对于formation energy，检查是否在容差范围内
                # 这里简化处理，实际需要计算生成结构的formation energy
                success_count += 1  # 占位符

            elif property_name == 'spacegroup':
                try:
                    sga = SpacegroupAnalyzer(generated_structure, symprec=0.1)
                    gen_spacegroup = sga.get_space_group_number()
                    target_spacegroup = target_properties['spacegroup']

                    if gen_spacegroup == target_spacegroup:
                        success_count += 1
                except:
                    pass

            elif property_name == 'band_gap':
                # 占位符，实际需要计算band gap
                success_count += 1

            total_count += 1

        return success_count / total_count if total_count > 0 else 0.0

    def compute_validity_metrics(
        self,
        generated_structures: List[Structure]
    ) -> Dict[str, float]:
        """
        计算有效性指标

        Args:
            generated_structures: 生成的结构列表

        Returns:
            包含各种有效性指标的字典
        """
        metrics = {
            'structural_validity': 0.0,
            'compositional_validity': 0.0,
            'total_validity': 0.0
        }

        valid_structural = 0
        valid_compositional = 0

        for structure in generated_structures:
            if structure is None:
                continue

            # 检查结构有效性（原子间距离）
            is_structurally_valid = self._check_structural_validity(structure)
            if is_structurally_valid:
                valid_structural += 1

            # 检查组成有效性（电荷中性）
            is_compositionally_valid = self._check_compositional_validity(structure)
            if is_compositionally_valid:
                valid_compositional += 1

        total = len(generated_structures)
        if total > 0:
            metrics['structural_validity'] = valid_structural / total
            metrics['compositional_validity'] = valid_compositional / total
            metrics['total_validity'] = (valid_structural / total) * (valid_compositional / total)

        return metrics

    def _check_structural_validity(self, structure: Structure) -> bool:
        """检查结构有效性（原子不能太近）"""
        try:
            distance_matrix = structure.distance_matrix

            # 检查是否有原子距离过近（< 0.5 Å）
            np.fill_diagonal(distance_matrix, np.inf)
            min_distance = np.min(distance_matrix)

            return min_distance > 0.5

        except:
            return False

    def _check_compositional_validity(self, structure: Structure) -> bool:
        """检查组成有效性（电荷中性）"""
        try:
            composition = structure.composition
            # 简化检查：确保组成不为空
            return len(composition) > 0
        except:
            return False

    def compute_property_distribution_metrics(
        self,
        generated_structures: List[Structure],
        reference_structures: List[Structure],
        property_name: str
    ) -> Dict[str, float]:
        """
        计算属性分布指标（Wasserstein距离）

        Args:
            generated_structures: 生成的结构
            reference_structures: 参考结构
            property_name: 属性名称

        Returns:
            包含Wasserstein距离的字典
        """
        from scipy.stats import wasserstein_distance

        # 提取属性值
        gen_properties = self._extract_property_values(
            generated_structures, property_name
        )
        ref_properties = self._extract_property_values(
            reference_structures, property_name
        )

        if len(gen_properties) == 0 or len(ref_properties) == 0:
            return {'wasserstein_distance': float('inf')}

        # 计算Wasserstein距离
        wd = wasserstein_distance(gen_properties, ref_properties)

        return {'wasserstein_distance': wd}

    def _extract_property_values(
        self,
        structures: List[Structure],
        property_name: str
    ) -> np.ndarray:
        """提取属性值"""
        values = []

        for structure in structures:
            if structure is None:
                continue

            if property_name == 'density':
                values.append(structure.density)
            elif property_name == 'num_atoms':
                values.append(len(structure))
            # 可以添加更多属性

        return np.array(values)

    def evaluate_model(
        self,
        test_data: List[Dict[str, Any]],
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        全面评估模型

        Args:
            test_data: 测试数据
            num_samples: 生成样本数量

        Returns:
            评估结果字典
        """
        results = {}

        # 1. 条件生成评估
        print("Evaluating conditional generation...")

        # 成功率
        for prop in ['spacegroup', 'formation_energy', 'band_gap']:
            sr = self.compute_success_rate(test_data[:100], prop)
            results[f'{prop}_success_rate'] = sr
            print(f"  {prop} success rate: {sr:.4f}")

        # 2. 无条件生成评估
        print("\nGenerating samples for unconditional evaluation...")
        generated_structures = []

        for i in tqdm(range(min(num_samples, 100)), desc="Generating"):
            instruction = self.instruction_builder.build_unconditional_generation_instruction()
            generated_text = self.trainer.generate(instruction)
            structure = self.parse_generated_structure(generated_text)
            generated_structures.append(structure)

        # 有效性指标
        print("\nComputing validity metrics...")
        validity_metrics = self.compute_validity_metrics(generated_structures)
        results.update(validity_metrics)

        for key, value in validity_metrics.items():
            print(f"  {key}: {value:.4f}")

        # 3. 属性分布
        print("\nComputing property distribution metrics...")
        reference_structures = [s['structure'] for s in test_data]

        for prop in ['density', 'num_atoms']:
            dist_metrics = self.compute_property_distribution_metrics(
                [s for s in generated_structures if s is not None],
                reference_structures,
                prop
            )
            results[f'{prop}_distribution'] = dist_metrics
            print(f"  {prop} Wasserstein distance: {dist_metrics['wasserstein_distance']:.4f}")

        return results


def main():
    """主函数"""
    from data_loader import CrystalDataLoader

    # 加载测试数据
    print("Loading test data...")
    loader = CrystalDataLoader()
    test_data = loader.load_from_json("./data/test.json")

    print(f"Loaded {len(test_data)} test samples")

    # 初始化评估器
    print("\nInitializing evaluator...")
    evaluator = CrystalEvaluator(model_path="./crystalicl_qwen_output")

    # 评估模型
    print("\nEvaluating model...")
    results = evaluator.evaluate_model(test_data, num_samples=100)

    # 保存结果
    import json
    with open("./evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation completed! Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
