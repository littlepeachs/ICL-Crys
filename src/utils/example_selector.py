"""
Example Selection Strategies for Few-shot Learning
实现论文中的三种示例选择策略
"""

import numpy as np
from typing import List, Dict, Any
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import random


class ExampleSelector:
    """示例选择器"""

    def __init__(self):
        self.structure_matcher = StructureMatcher()

    def condition_based_selection(
        self,
        dataset: List[Dict[str, Any]],
        target_properties: Dict[str, Any],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        基于条件的选择策略
        选择满足目标属性条件的晶体

        Args:
            dataset: 数据集
            target_properties: 目标属性
            k: 选择数量

        Returns:
            选中的示例列表
        """
        filtered = []

        for sample in dataset:
            properties = sample.get('properties', {})
            match = True

            # 检查化学式
            if 'chemical_formula' in target_properties:
                if properties.get('chemical_formula') != target_properties['chemical_formula']:
                    match = False

            # 检查空间群
            if 'spacegroup' in target_properties and match:
                if properties.get('spacegroup') != target_properties['spacegroup']:
                    match = False

            # 对于连续属性（如band gap），使用范围匹配
            if 'band_gap' in target_properties and match:
                target_bg = target_properties['band_gap']
                sample_bg = properties.get('band_gap', None)
                if sample_bg is not None:
                    # 允许0.5 eV的误差
                    if abs(sample_bg - target_bg) > 0.5:
                        match = False

            if match:
                filtered.append(sample)

        # 如果匹配的样本不足k个，随机补充
        if len(filtered) < k:
            remaining = [s for s in dataset if s not in filtered]
            filtered.extend(random.sample(remaining, min(k - len(filtered), len(remaining))))

        return random.sample(filtered, min(k, len(filtered)))

    def structure_based_selection(
        self,
        dataset: List[Dict[str, Any]],
        anchor_structure: Structure = None,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        基于结构的选择策略
        选择结构相似的晶体

        Args:
            dataset: 数据集
            anchor_structure: 锚点结构（如果为None则随机选择）
            k: 选择数量

        Returns:
            选中的示例列表
        """
        if anchor_structure is None:
            anchor_sample = random.choice(dataset)
            anchor_structure = anchor_sample['structure']

        # 计算结构相似度（使用简化的指纹距离）
        similarities = []
        for sample in dataset:
            structure = sample['structure']
            # 使用晶格参数和组成作为简单的相似度度量
            similarity = self._compute_structure_similarity(anchor_structure, structure)
            similarities.append((sample, similarity))

        # 按相似度排序并选择top-k
        similarities.sort(key=lambda x: x[1])
        selected = [s[0] for s in similarities[:k]]

        return selected

    def condition_structure_based_selection(
        self,
        dataset: List[Dict[str, Any]],
        target_properties: Dict[str, Any],
        anchor_structure: Structure = None,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        基于条件和结构的混合选择策略
        先根据属性过滤，再根据结构相似度选择

        Args:
            dataset: 数据集
            target_properties: 目标属性
            anchor_structure: 锚点结构
            k: 选择数量

        Returns:
            选中的示例列表
        """
        # 第一步：基于条件过滤
        filtered = []
        for sample in dataset:
            properties = sample.get('properties', {})

            # 检查化学式（如果指定）
            if 'chemical_formula' in target_properties:
                formula = target_properties['chemical_formula']
                # 提取元素类型（忽略具体化学计量比）
                if properties.get('chemical_formula', '').replace('2', '').replace('3', '') != \
                   formula.replace('2', '').replace('3', ''):
                    continue

            # 检查空间群（如果指定）
            if 'spacegroup' in target_properties:
                if properties.get('spacegroup') != target_properties['spacegroup']:
                    continue

            filtered.append(sample)

        # 如果过滤后样本不足，使用全部数据集
        if len(filtered) < k:
            filtered = dataset

        # 第二步：基于结构相似度选择
        if anchor_structure is None:
            anchor_structure = random.choice(filtered)['structure']

        similarities = []
        for sample in filtered:
            structure = sample['structure']
            similarity = self._compute_structure_similarity(anchor_structure, structure)
            similarities.append((sample, similarity))

        # 按相似度排序并选择top-k
        similarities.sort(key=lambda x: x[1])
        selected = [s[0] for s in similarities[:k]]

        return selected

    def _compute_structure_similarity(self, struct1: Structure, struct2: Structure) -> float:
        """
        计算两个结构的相似度（欧氏距离）

        使用CrystalNN指纹的简化版本
        """
        # 特征1: 晶格参数
        lattice1 = struct1.lattice
        lattice2 = struct2.lattice

        lattice_diff = np.linalg.norm([
            lattice1.a - lattice2.a,
            lattice1.b - lattice2.b,
            lattice1.c - lattice2.c,
            lattice1.alpha - lattice2.alpha,
            lattice1.beta - lattice2.beta,
            lattice1.gamma - lattice2.gamma
        ])

        # 特征2: 组成差异
        comp1 = struct1.composition.fractional_composition
        comp2 = struct2.composition.fractional_composition

        elements1 = set(comp1.elements)
        elements2 = set(comp2.elements)

        # Jaccard距离
        composition_diff = 1.0 - len(elements1 & elements2) / len(elements1 | elements2)

        # 特征3: 原子数差异
        natoms_diff = abs(len(struct1) - len(struct2)) / max(len(struct1), len(struct2))

        # 综合相似度
        similarity = lattice_diff + 10 * composition_diff + 5 * natoms_diff

        return similarity

    def random_selection(
        self,
        dataset: List[Dict[str, Any]],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """随机选择策略（基线）"""
        return random.sample(dataset, min(k, len(dataset)))


def test_example_selector():
    """测试示例选择器"""
    from pymatgen.core import Lattice, Structure

    # 创建测试数据集
    dataset = []

    # NaCl类型结构
    for i in range(5):
        lattice = Lattice.cubic(5.64 + i * 0.1)
        structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])
        dataset.append({
            'structure': structure,
            'properties': {
                'chemical_formula': 'NaCl',
                'spacegroup': 225,
                'formation_energy': -0.5 - i * 0.1,
                'band_gap': 5.0 + i * 0.2
            }
        })

    # MgO类型结构
    for i in range(5):
        lattice = Lattice.cubic(4.21 + i * 0.1)
        structure = Structure(lattice, ['Mg', 'O'], [[0, 0, 0], [0.5, 0.5, 0.5]])
        dataset.append({
            'structure': structure,
            'properties': {
                'chemical_formula': 'MgO',
                'spacegroup': 225,
                'formation_energy': -0.6 - i * 0.1,
                'band_gap': 7.8 + i * 0.2
            }
        })

    selector = ExampleSelector()

    # 测试条件选择
    print("Condition-based Selection:")
    target_props = {'chemical_formula': 'NaCl', 'spacegroup': 225}
    selected = selector.condition_based_selection(dataset, target_props, k=3)
    for i, sample in enumerate(selected, 1):
        print(f"  Example {i}: {sample['properties']['chemical_formula']}")
    print()

    # 测试结构选择
    print("Structure-based Selection:")
    anchor = dataset[0]['structure']
    selected = selector.structure_based_selection(dataset, anchor, k=3)
    for i, sample in enumerate(selected, 1):
        print(f"  Example {i}: {sample['properties']['chemical_formula']}")
    print()

    # 测试混合选择
    print("Condition-Structure based Selection:")
    selected = selector.condition_structure_based_selection(
        dataset, target_props, anchor, k=3
    )
    for i, sample in enumerate(selected, 1):
        print(f"  Example {i}: {sample['properties']['chemical_formula']}")


if __name__ == "__main__":
    test_example_selector()
