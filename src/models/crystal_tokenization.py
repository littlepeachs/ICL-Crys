"""
Space-group based Crystal Tokenization (SGS)
将晶体结构转换为基于空间群的文本表示
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class CrystalTokenizer:
    """基于空间群的晶体token化器"""

    def __init__(self):
        pass

    def get_wyckoff_positions(self, structure: Structure) -> Dict:
        """获取Wyckoff位置信息"""
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            symmetrized_structure = sga.get_symmetrized_structure()

            wyckoff_dict = {}
            for i, equiv_sites in enumerate(symmetrized_structure.equivalent_sites):
                site = equiv_sites[0]
                wyckoff_letter = symmetrized_structure.wyckoff_symbols[i]
                element = site.species_string

                if element not in wyckoff_dict:
                    wyckoff_dict[element] = []

                wyckoff_dict[element].append({
                    'wyckoff': wyckoff_letter,
                    'coords': site.frac_coords.tolist()
                })

            return wyckoff_dict
        except Exception as e:
            return {}

    def structure_to_sgs(self, structure: Structure) -> str:
        """将晶体结构转换为SGS格式文本"""
        try:
            # 获取空间群信息
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            spacegroup = sga.get_space_group_symbol()
            spacegroup_number = sga.get_space_group_number()

            # 获取晶格参数
            lattice = structure.lattice
            a, b, c = lattice.a, lattice.b, lattice.c
            alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

            # 获取Wyckoff位置
            wyckoff_dict = self.get_wyckoff_positions(structure)

            # 构建SGS格式文本
            sgs_text = f"{spacegroup}\n"
            sgs_text += f"{a:.3f} {b:.3f} {c:.3f}\n"
            sgs_text += f"{alpha:.1f} {beta:.1f} {gamma:.1f}\n"

            # 添加元素和Wyckoff位置
            for element, positions in wyckoff_dict.items():
                for pos in positions:
                    wyckoff = pos['wyckoff']
                    coords = pos['coords']
                    sgs_text += f"{element}\n"
                    sgs_text += f"{coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}\n"

            return sgs_text.strip()

        except Exception as e:
            # 如果空间群分析失败，使用XYZ格式作为后备
            return self.structure_to_xyz(structure)

    def structure_to_xyz(self, structure: Structure) -> str:
        """将晶体结构转换为XYZ格式（后备方案）"""
        lattice = structure.lattice
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

        xyz_text = f"Lattice: {a:.3f} {b:.3f} {c:.3f}\n"
        xyz_text += f"Angles: {alpha:.1f} {beta:.1f} {gamma:.1f}\n"

        for site in structure:
            element = site.species_string
            coords = site.frac_coords
            xyz_text += f"{element} {coords[0]:.3f} {coords[1]:.3f} {coords[2]:.3f}\n"

        return xyz_text.strip()

    def tokenize(self, structure: Structure, use_sgs: bool = True) -> str:
        """
        将晶体结构token化

        Args:
            structure: pymatgen Structure对象
            use_sgs: 是否使用SGS格式（默认True）

        Returns:
            token化后的文本表示
        """
        if use_sgs:
            return self.structure_to_sgs(structure)
        else:
            return self.structure_to_xyz(structure)


def test_tokenizer():
    """测试token化器"""
    # 创建一个简单的NaCl结构
    lattice = [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]
    species = ['Na', 'Cl']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(lattice, species, coords)

    tokenizer = CrystalTokenizer()

    # SGS格式
    sgs_text = tokenizer.tokenize(structure, use_sgs=True)
    print("SGS Format:")
    print(sgs_text)
    print("\n" + "="*50 + "\n")

    # XYZ格式
    xyz_text = tokenizer.tokenize(structure, use_sgs=False)
    print("XYZ Format:")
    print(xyz_text)


if __name__ == "__main__":
    test_tokenizer()
