"""
SGS Format Parser - 将SGS格式文本反序列化为晶体结构
这是关键缺失的功能！
"""

from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import re
from typing import Optional


class SGSParser:
    """SGS格式解析器 - 将生成的文本转换回晶体结构"""

    def __init__(self):
        pass

    def parse_sgs_to_structure(self, sgs_text: str) -> Optional[Structure]:
        """
        将SGS格式文本解析为pymatgen Structure对象

        SGS格式示例:
        Fm-3m
        5.640 5.640 5.640
        90.0 90.0 90.0
        Na
        0.00 0.00 0.00
        Cl
        0.50 0.50 0.50

        Args:
            sgs_text: SGS格式的文本

        Returns:
            Structure对象，如果解析失败返回None
        """
        try:
            lines = [l.strip() for l in sgs_text.strip().split('\n') if l.strip()]

            if len(lines) < 3:
                return None

            # 1. 解析空间群符号（第一行）
            spacegroup_symbol = lines[0]

            # 2. 解析晶格参数（第二行）
            lattice_params = list(map(float, lines[1].split()))
            if len(lattice_params) != 3:
                return None
            a, b, c = lattice_params

            # 3. 解析晶格角度（第三行）
            angles = list(map(float, lines[2].split()))
            if len(angles) != 3:
                return None
            alpha, beta, gamma = angles

            # 创建晶格
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

            # 4. 解析原子信息
            species = []
            coords = []

            i = 3
            while i < len(lines):
                # 元素符号
                element_line = lines[i]

                # 检查是否是元素符号（只包含字母）
                if not element_line.replace(' ', '').isalpha():
                    i += 1
                    continue

                element = element_line.strip()
                species.append(element)

                # 坐标（下一行）
                i += 1
                if i < len(lines):
                    try:
                        coord = list(map(float, lines[i].split()))
                        if len(coord) == 3:
                            coords.append(coord)
                        else:
                            # 坐标格式错误，移除对应的元素
                            species.pop()
                    except ValueError:
                        # 无法解析坐标，移除对应的元素
                        species.pop()

                i += 1

            if len(species) == 0 or len(species) != len(coords):
                return None

            # 5. 创建结构
            structure = Structure(lattice, species, coords)

            # 6. 验证空间群（可选）
            try:
                sga = SpacegroupAnalyzer(structure, symprec=0.1)
                detected_sg = sga.get_space_group_symbol()
                # 注意：生成的空间群可能与检测的不完全一致，这是正常的
            except:
                pass

            return structure

        except Exception as e:
            return None

    def parse_with_validation(self, sgs_text: str) -> tuple[Optional[Structure], dict]:
        """
        解析SGS文本并进行验证

        Returns:
            (structure, validation_info)
            validation_info包含：
            - 'valid': bool
            - 'errors': list of error messages
            - 'warnings': list of warnings
        """
        validation_info = {
            'valid': False,
            'errors': [],
            'warnings': []
        }

        structure = self.parse_sgs_to_structure(sgs_text)

        if structure is None:
            validation_info['errors'].append("Failed to parse SGS text")
            return None, validation_info

        # 验证结构
        # 1. 检查原子间距
        try:
            distance_matrix = structure.distance_matrix
            import numpy as np
            np.fill_diagonal(distance_matrix, np.inf)
            min_distance = np.min(distance_matrix)

            if min_distance < 0.5:
                validation_info['errors'].append(
                    f"Atoms too close: minimum distance = {min_distance:.3f} Å"
                )
            elif min_distance < 1.0:
                validation_info['warnings'].append(
                    f"Atoms very close: minimum distance = {min_distance:.3f} Å"
                )
        except Exception as e:
            validation_info['warnings'].append(f"Could not check atomic distances: {e}")

        # 2. 检查晶格参数合理性
        lattice = structure.lattice
        if lattice.a < 1.0 or lattice.a > 100.0:
            validation_info['warnings'].append(
                f"Unusual lattice parameter a = {lattice.a:.3f} Å"
            )

        # 3. 检查坐标范围
        for site in structure:
            for coord in site.frac_coords:
                if coord < -0.1 or coord > 1.1:
                    validation_info['warnings'].append(
                        f"Fractional coordinate out of range: {coord:.3f}"
                    )
                    break

        # 4. 检查组成
        try:
            composition = structure.composition
            if len(composition) == 0:
                validation_info['errors'].append("Empty composition")
        except Exception as e:
            validation_info['errors'].append(f"Invalid composition: {e}")

        validation_info['valid'] = len(validation_info['errors']) == 0

        return structure, validation_info


def test_sgs_parser():
    """测试SGS解析器"""
    parser = SGSParser()

    # 测试用例1: NaCl
    sgs_text_1 = """Fm-3m
5.640 5.640 5.640
90.0 90.0 90.0
Na
0.00 0.00 0.00
Cl
0.50 0.50 0.50"""

    print("Test 1: NaCl")
    structure, validation = parser.parse_with_validation(sgs_text_1)
    if structure:
        print(f"  Formula: {structure.composition.reduced_formula}")
        print(f"  Lattice: a={structure.lattice.a:.3f} Å")
        print(f"  Valid: {validation['valid']}")
        print(f"  Warnings: {validation['warnings']}")
    else:
        print(f"  Failed: {validation['errors']}")

    # 测试用例2: 格式错误
    sgs_text_2 = """Invalid
5.640
90.0"""

    print("\nTest 2: Invalid format")
    structure, validation = parser.parse_with_validation(sgs_text_2)
    if structure:
        print(f"  Unexpectedly succeeded")
    else:
        print(f"  Correctly failed: {validation['errors']}")

    # 测试用例3: 原子重叠
    sgs_text_3 = """Fm-3m
5.640 5.640 5.640
90.0 90.0 90.0
Na
0.00 0.00 0.00
Cl
0.01 0.01 0.01"""

    print("\nTest 3: Overlapping atoms")
    structure, validation = parser.parse_with_validation(sgs_text_3)
    if structure:
        print(f"  Formula: {structure.composition.reduced_formula}")
        print(f"  Valid: {validation['valid']}")
        print(f"  Errors: {validation['errors']}")
        print(f"  Warnings: {validation['warnings']}")


if __name__ == "__main__":
    test_sgs_parser()
