"""
Crystal Structure Validation Tools
晶体结构验证和修正工具
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StructureValidator:
    """晶体结构验证器"""

    def __init__(self):
        self.structure_matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    def validate_structure(self, structure: Structure) -> Dict[str, any]:
        """
        全面验证晶体结构

        Returns:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'metrics': Dict[str, float]
            }
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }

        # 1. 检查原子间距
        distance_check = self._check_atomic_distances(structure)
        result['metrics']['min_distance'] = distance_check['min_distance']
        if not distance_check['valid']:
            result['valid'] = False
            result['errors'].extend(distance_check['errors'])
        result['warnings'].extend(distance_check['warnings'])

        # 2. 检查晶格参数
        lattice_check = self._check_lattice_parameters(structure)
        result['metrics'].update(lattice_check['metrics'])
        if not lattice_check['valid']:
            result['valid'] = False
            result['errors'].extend(lattice_check['errors'])
        result['warnings'].extend(lattice_check['warnings'])

        # 3. 检查坐标范围
        coord_check = self._check_coordinates(structure)
        if not coord_check['valid']:
            result['warnings'].extend(coord_check['warnings'])

        # 4. 检查组成
        comp_check = self._check_composition(structure)
        result['metrics']['num_atoms'] = comp_check['num_atoms']
        result['metrics']['num_elements'] = comp_check['num_elements']
        if not comp_check['valid']:
            result['valid'] = False
            result['errors'].extend(comp_check['errors'])

        # 5. 检查对称性
        symmetry_check = self._check_symmetry(structure)
        result['metrics']['spacegroup'] = symmetry_check.get('spacegroup', 0)
        result['warnings'].extend(symmetry_check.get('warnings', []))

        # 6. 检查密度
        density_check = self._check_density(structure)
        result['metrics']['density'] = density_check['density']
        result['warnings'].extend(density_check['warnings'])

        return result

    def _check_atomic_distances(self, structure: Structure) -> Dict:
        """检查原子间距"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'min_distance': 0.0
        }

        try:
            distance_matrix = structure.distance_matrix
            np.fill_diagonal(distance_matrix, np.inf)
            min_distance = np.min(distance_matrix)
            result['min_distance'] = min_distance

            if min_distance < 0.5:
                result['valid'] = False
                result['errors'].append(
                    f"Atoms too close: minimum distance = {min_distance:.3f} Å (< 0.5 Å)"
                )
            elif min_distance < 1.0:
                result['warnings'].append(
                    f"Atoms very close: minimum distance = {min_distance:.3f} Å"
                )

        except Exception as e:
            result['warnings'].append(f"Could not check atomic distances: {e}")

        return result

    def _check_lattice_parameters(self, structure: Structure) -> Dict:
        """检查晶格参数合理性"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }

        lattice = structure.lattice
        result['metrics']['lattice_a'] = lattice.a
        result['metrics']['lattice_b'] = lattice.b
        result['metrics']['lattice_c'] = lattice.c
        result['metrics']['volume'] = lattice.volume

        # 检查晶格参数范围
        for param_name, param_value in [('a', lattice.a), ('b', lattice.b), ('c', lattice.c)]:
            if param_value < 1.0:
                result['errors'].append(
                    f"Lattice parameter {param_name} too small: {param_value:.3f} Å"
                )
                result['valid'] = False
            elif param_value > 100.0:
                result['warnings'].append(
                    f"Lattice parameter {param_name} very large: {param_value:.3f} Å"
                )

        # 检查角度
        for angle_name, angle_value in [
            ('alpha', lattice.alpha),
            ('beta', lattice.beta),
            ('gamma', lattice.gamma)
        ]:
            if angle_value < 30.0 or angle_value > 150.0:
                result['warnings'].append(
                    f"Unusual lattice angle {angle_name}: {angle_value:.1f}°"
                )

        # 检查体积
        if lattice.volume < 10.0:
            result['warnings'].append(f"Very small unit cell volume: {lattice.volume:.2f} ų")
        elif lattice.volume > 10000.0:
            result['warnings'].append(f"Very large unit cell volume: {lattice.volume:.2f} ų")

        return result

    def _check_coordinates(self, structure: Structure) -> Dict:
        """检查坐标范围"""
        result = {
            'valid': True,
            'warnings': []
        }

        for i, site in enumerate(structure):
            for j, coord in enumerate(site.frac_coords):
                if coord < -0.1 or coord > 1.1:
                    result['warnings'].append(
                        f"Site {i} coordinate {j} out of range: {coord:.3f}"
                    )

        return result

    def _check_composition(self, structure: Structure) -> Dict:
        """检查组成"""
        result = {
            'valid': True,
            'errors': [],
            'num_atoms': 0,
            'num_elements': 0
        }

        try:
            composition = structure.composition
            result['num_atoms'] = len(structure)
            result['num_elements'] = len(set(structure.species))

            if len(composition) == 0:
                result['valid'] = False
                result['errors'].append("Empty composition")

            # 检查是否有负的原子数
            for element, amount in composition.items():
                if amount <= 0:
                    result['valid'] = False
                    result['errors'].append(
                        f"Invalid amount for {element}: {amount}"
                    )

        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Invalid composition: {e}")

        return result

    def _check_symmetry(self, structure: Structure) -> Dict:
        """检查对称性"""
        result = {
            'warnings': []
        }

        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            result['spacegroup'] = sga.get_space_group_number()
            result['crystal_system'] = sga.get_crystal_system()
            result['point_group'] = sga.get_point_group_symbol()
        except Exception as e:
            result['warnings'].append(f"Could not analyze symmetry: {e}")
            result['spacegroup'] = 0

        return result

    def _check_density(self, structure: Structure) -> Dict:
        """检查密度"""
        result = {
            'warnings': [],
            'density': 0.0
        }

        try:
            density = structure.density
            result['density'] = density

            if density < 0.1:
                result['warnings'].append(f"Very low density: {density:.3f} g/cm³")
            elif density > 30.0:
                result['warnings'].append(f"Very high density: {density:.3f} g/cm³")

        except Exception as e:
            result['warnings'].append(f"Could not calculate density: {e}")

        return result

    def fix_structure(self, structure: Structure) -> Tuple[Structure, Dict]:
        """
        尝试修正结构中的问题

        Returns:
            (fixed_structure, fix_info)
        """
        fix_info = {
            'fixed': False,
            'changes': []
        }

        fixed_structure = structure.copy()

        # 1. 归一化坐标到[0, 1)
        for i, site in enumerate(fixed_structure):
            original_coords = site.frac_coords.copy()
            normalized_coords = original_coords % 1.0

            if not np.allclose(original_coords, normalized_coords):
                fixed_structure.replace(i, site.species, normalized_coords)
                fix_info['fixed'] = True
                fix_info['changes'].append(
                    f"Normalized coordinates for site {i}"
                )

        # 2. 移除重复原子
        # （这里简化处理，实际可能需要更复杂的逻辑）

        return fixed_structure, fix_info


class StructureComparator:
    """结构比较器"""

    def __init__(self):
        self.structure_matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    def compare_structures(
        self,
        structure1: Structure,
        structure2: Structure
    ) -> Dict[str, any]:
        """
        比较两个结构

        Returns:
            {
                'match': bool,
                'rms_distance': float,
                'composition_match': bool,
                'spacegroup_match': bool
            }
        """
        result = {
            'match': False,
            'rms_distance': None,
            'composition_match': False,
            'spacegroup_match': False
        }

        # 1. 结构匹配
        try:
            match = self.structure_matcher.fit(structure1, structure2)
            result['match'] = match

            if match:
                rms = self.structure_matcher.get_rms_dist(structure1, structure2)
                result['rms_distance'] = rms[0] if rms else None
        except:
            pass

        # 2. 组成匹配
        try:
            comp1 = structure1.composition.reduced_formula
            comp2 = structure2.composition.reduced_formula
            result['composition_match'] = (comp1 == comp2)
        except:
            pass

        # 3. 空间群匹配
        try:
            sga1 = SpacegroupAnalyzer(structure1, symprec=0.1)
            sga2 = SpacegroupAnalyzer(structure2, symprec=0.1)
            sg1 = sga1.get_space_group_number()
            sg2 = sga2.get_space_group_number()
            result['spacegroup_match'] = (sg1 == sg2)
        except:
            pass

        return result


def test_validator():
    """测试验证器"""
    validator = StructureValidator()

    # 测试1: 正常结构
    print("Test 1: Valid structure (NaCl)")
    lattice = Lattice.cubic(5.64)
    structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

    result = validator.validate_structure(structure)
    print(f"  Valid: {result['valid']}")
    print(f"  Errors: {result['errors']}")
    print(f"  Warnings: {result['warnings']}")
    print(f"  Metrics: {result['metrics']}")

    # 测试2: 原子重叠
    print("\nTest 2: Overlapping atoms")
    structure2 = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.01, 0.01, 0.01]])

    result2 = validator.validate_structure(structure2)
    print(f"  Valid: {result2['valid']}")
    print(f"  Errors: {result2['errors']}")

    # 测试3: 修正结构
    print("\nTest 3: Fix structure with out-of-range coordinates")
    structure3 = Structure(lattice, ['Na', 'Cl'], [[1.5, 0, 0], [0.5, 0.5, 0.5]])

    fixed, fix_info = validator.fix_structure(structure3)
    print(f"  Fixed: {fix_info['fixed']}")
    print(f"  Changes: {fix_info['changes']}")


if __name__ == "__main__":
    test_validator()
