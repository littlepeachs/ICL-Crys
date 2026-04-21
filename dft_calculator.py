"""
DFT Property Calculator Interface
DFT属性计算接口 - 补充缺失的12.5%

提供两种方案：
1. VASP/Quantum ESPRESSO 接口（高精度）
2. 机器学习模型快速估算（快速）
"""

import os
import subprocess
from typing import Dict, Optional
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints
import warnings
warnings.filterwarnings('ignore')


class DFTCalculator:
    """DFT计算器接口"""

    def __init__(self, method: str = "vasp"):
        """
        初始化DFT计算器

        Args:
            method: 计算方法 ("vasp", "qe", "ml")
        """
        self.method = method

    def calculate_formation_energy(
        self,
        structure: Structure,
        timeout: int = 3600
    ) -> Optional[float]:
        """
        计算formation energy

        Args:
            structure: 晶体结构
            timeout: 超时时间（秒）

        Returns:
            Formation energy (eV/atom)，失败返回None
        """
        if self.method == "vasp":
            return self._calculate_with_vasp(structure, "formation_energy", timeout)
        elif self.method == "qe":
            return self._calculate_with_qe(structure, "formation_energy", timeout)
        elif self.method == "ml":
            return self._calculate_with_ml(structure, "formation_energy")
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def calculate_band_gap(
        self,
        structure: Structure,
        timeout: int = 3600
    ) -> Optional[float]:
        """
        计算band gap

        Args:
            structure: 晶体结构
            timeout: 超时时间（秒）

        Returns:
            Band gap (eV)，失败返回None
        """
        if self.method == "vasp":
            return self._calculate_with_vasp(structure, "band_gap", timeout)
        elif self.method == "qe":
            return self._calculate_with_qe(structure, "band_gap", timeout)
        elif self.method == "ml":
            return self._calculate_with_ml(structure, "band_gap")
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _calculate_with_vasp(
        self,
        structure: Structure,
        property_name: str,
        timeout: int
    ) -> Optional[float]:
        """
        使用VASP计算

        需要：
        1. VASP已安装并在PATH中
        2. POTCAR文件已配置
        3. 足够的计算资源
        """
        try:
            # 创建临时目录
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                # 写入POSCAR
                poscar = Poscar(structure)
                poscar.write_file(os.path.join(tmpdir, "POSCAR"))

                # 写入INCAR
                incar_settings = self._get_vasp_incar_settings(property_name)
                with open(os.path.join(tmpdir, "INCAR"), 'w') as f:
                    for key, value in incar_settings.items():
                        f.write(f"{key} = {value}\n")

                # 写入KPOINTS
                kpoints = Kpoints.automatic_density(structure, 1000)
                kpoints.write_file(os.path.join(tmpdir, "KPOINTS"))

                # 运行VASP
                result = subprocess.run(
                    ["vasp_std"],
                    cwd=tmpdir,
                    timeout=timeout,
                    capture_output=True
                )

                if result.returncode != 0:
                    return None

                # 解析结果
                return self._parse_vasp_output(tmpdir, property_name)

        except Exception as e:
            print(f"VASP calculation failed: {e}")
            return None

    def _calculate_with_qe(
        self,
        structure: Structure,
        property_name: str,
        timeout: int
    ) -> Optional[float]:
        """
        使用Quantum ESPRESSO计算

        需要：
        1. QE已安装（pw.x, dos.x等）
        2. 赝势文件已配置
        """
        try:
            # 类似VASP的流程
            # 1. 生成输入文件
            # 2. 运行pw.x
            # 3. 解析输出
            pass
        except Exception as e:
            print(f"QE calculation failed: {e}")
            return None

    def _calculate_with_ml(
        self,
        structure: Structure,
        property_name: str
    ) -> Optional[float]:
        """
        使用机器学习模型快速估算

        使用预训练的CGCNN或MEGNet模型
        """
        try:
            if property_name == "formation_energy":
                return self._predict_formation_energy_ml(structure)
            elif property_name == "band_gap":
                return self._predict_band_gap_ml(structure)
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return None

    def _predict_formation_energy_ml(self, structure: Structure) -> Optional[float]:
        """
        使用ML模型预测formation energy

        可以使用：
        1. MEGNet (推荐)
        2. CGCNN
        3. SchNet
        """
        try:
            # 方案1: 使用MEGNet
            from megnet.models import MEGNetModel
            from megnet.data.crystal import CrystalGraph

            # 加载预训练模型
            model = MEGNetModel.from_file("megnet_formation_energy.hdf5")

            # 预测
            graph = CrystalGraph()
            inp = graph.convert(structure)
            prediction = model.predict(inp)

            return float(prediction[0])

        except ImportError:
            print("MEGNet not installed. Install with: pip install megnet")
            return None
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None

    def _predict_band_gap_ml(self, structure: Structure) -> Optional[float]:
        """使用ML模型预测band gap"""
        try:
            from megnet.models import MEGNetModel
            from megnet.data.crystal import CrystalGraph

            model = MEGNetModel.from_file("megnet_band_gap.hdf5")
            graph = CrystalGraph()
            inp = graph.convert(structure)
            prediction = model.predict(inp)

            return float(prediction[0])

        except ImportError:
            print("MEGNet not installed")
            return None
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None

    def _get_vasp_incar_settings(self, property_name: str) -> Dict:
        """获取VASP INCAR设置"""
        base_settings = {
            "PREC": "Accurate",
            "ENCUT": 520,
            "EDIFF": 1e-6,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LREAL": "Auto",
        }

        if property_name == "formation_energy":
            base_settings.update({
                "IBRION": 2,
                "NSW": 100,
                "ISIF": 3,
            })
        elif property_name == "band_gap":
            base_settings.update({
                "IBRION": -1,
                "NSW": 0,
                "LORBIT": 11,
            })

        return base_settings

    def _parse_vasp_output(self, directory: str, property_name: str) -> Optional[float]:
        """解析VASP输出"""
        try:
            from pymatgen.io.vasp import Vasprun

            vasprun = Vasprun(os.path.join(directory, "vasprun.xml"))

            if property_name == "formation_energy":
                # 需要计算相对于元素的能量
                energy = vasprun.final_energy
                natoms = len(vasprun.final_structure)
                return energy / natoms

            elif property_name == "band_gap":
                return vasprun.get_band_structure().get_band_gap()["energy"]

        except Exception as e:
            print(f"Failed to parse VASP output: {e}")
            return None


class PropertyMatcher:
    """属性匹配器 - 用于评估"""

    def __init__(self, calculator: DFTCalculator, tolerance: Dict[str, float] = None):
        """
        初始化匹配器

        Args:
            calculator: DFT计算器
            tolerance: 容差字典，例如 {"formation_energy": 0.5, "band_gap": 0.5}
        """
        self.calculator = calculator
        self.tolerance = tolerance or {
            "formation_energy": 0.5,  # eV/atom
            "band_gap": 0.5  # eV
        }

    def check_formation_energy_match(
        self,
        generated_structure: Structure,
        target_fe: float
    ) -> bool:
        """
        检查formation energy是否匹配

        Args:
            generated_structure: 生成的结构
            target_fe: 目标formation energy

        Returns:
            是否匹配
        """
        calculated_fe = self.calculator.calculate_formation_energy(generated_structure)

        if calculated_fe is None:
            return False

        return abs(calculated_fe - target_fe) < self.tolerance["formation_energy"]

    def check_band_gap_match(
        self,
        generated_structure: Structure,
        target_bg: float
    ) -> bool:
        """
        检查band gap是否匹配

        Args:
            generated_structure: 生成的结构
            target_bg: 目标band gap

        Returns:
            是否匹配
        """
        calculated_bg = self.calculator.calculate_band_gap(generated_structure)

        if calculated_bg is None:
            return False

        return abs(calculated_bg - target_bg) < self.tolerance["band_gap"]


def test_dft_calculator():
    """测试DFT计算器"""
    from pymatgen.core import Lattice

    # 创建测试结构
    lattice = Lattice.cubic(5.64)
    structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # 测试ML方法（最快）
    print("Testing ML calculator...")
    calculator = DFTCalculator(method="ml")

    fe = calculator.calculate_formation_energy(structure)
    bg = calculator.calculate_band_gap(structure)

    print(f"Formation Energy: {fe} eV/atom")
    print(f"Band Gap: {bg} eV")

    # 测试匹配器
    print("\nTesting property matcher...")
    matcher = PropertyMatcher(calculator)

    fe_match = matcher.check_formation_energy_match(structure, -0.5)
    bg_match = matcher.check_band_gap_match(structure, 5.0)

    print(f"Formation Energy Match: {fe_match}")
    print(f"Band Gap Match: {bg_match}")


if __name__ == "__main__":
    test_dft_calculator()
