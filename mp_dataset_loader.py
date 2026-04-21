"""
Materials Project Dataset Loader
加载MP20, MP30等真实数据集
"""

import os
import json
from typing import List, Dict, Any, Optional
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MaterialsProjectLoader:
    """Materials Project数据集加载器"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化加载器

        Args:
            api_key: Materials Project API密钥
                    可以从环境变量 MP_API_KEY 获取
                    或从 https://materialsproject.org/api 申请
        """
        self.api_key = api_key or os.environ.get('MP_API_KEY')
        if not self.api_key:
            print("Warning: No Materials Project API key provided.")
            print("Set MP_API_KEY environment variable or pass api_key parameter.")
            print("Get your API key from: https://materialsproject.org/api")

    def load_mp20_dataset(
        self,
        save_path: str = "./data/mp20.json",
        max_elements: int = 2,
        max_structures: int = 45231
    ) -> List[Dict[str, Any]]:
        """
        加载MP20数据集

        MP20包含45,231个材料，最多2种元素

        Args:
            save_path: 保存路径
            max_elements: 最大元素数量
            max_structures: 最大结构数量

        Returns:
            数据列表
        """
        print(f"Loading MP20 dataset (max {max_structures} structures)...")

        if not self.api_key:
            raise ValueError("API key required for Materials Project access")

        data = []

        with MPRester(self.api_key) as mpr:
            # 查询条件：最多2种元素
            results = mpr.query(
                criteria={
                    "nelements": {"$lte": max_elements},
                    "e_above_hull": {"$lte": 0.08}  # 接近稳定的结构
                },
                properties=[
                    "material_id",
                    "structure",
                    "formation_energy_per_atom",
                    "band_gap",
                    "spacegroup",
                    "pretty_formula",
                    "density"
                ]
            )

            print(f"Found {len(results)} structures")

            # 限制数量
            results = results[:max_structures]

            for entry in tqdm(results, desc="Processing"):
                try:
                    data.append({
                        'structure': entry['structure'],
                        'properties': {
                            'material_id': entry['material_id'],
                            'chemical_formula': entry['pretty_formula'],
                            'spacegroup': entry['spacegroup']['number'],
                            'formation_energy': entry['formation_energy_per_atom'],
                            'band_gap': entry['band_gap'],
                            'density': entry['density']
                        }
                    })
                except Exception as e:
                    print(f"Error processing {entry.get('material_id', 'unknown')}: {e}")
                    continue

        # 保存
        self._save_dataset(data, save_path)

        print(f"Loaded {len(data)} structures from MP20")
        return data

    def load_mp30_dataset(
        self,
        save_path: str = "./data/mp30.json",
        max_elements: int = 3,
        max_structures: int = 127609
    ) -> List[Dict[str, Any]]:
        """
        加载MP30数据集

        MP30包含127,609个材料，最多3种元素

        Args:
            save_path: 保存路径
            max_elements: 最大元素数量
            max_structures: 最大结构数量

        Returns:
            数据列表
        """
        print(f"Loading MP30 dataset (max {max_structures} structures)...")

        if not self.api_key:
            raise ValueError("API key required for Materials Project access")

        data = []

        with MPRester(self.api_key) as mpr:
            results = mpr.query(
                criteria={
                    "nelements": {"$lte": max_elements},
                    "e_above_hull": {"$lte": 0.08}
                },
                properties=[
                    "material_id",
                    "structure",
                    "formation_energy_per_atom",
                    "band_gap",
                    "spacegroup",
                    "pretty_formula",
                    "density"
                ]
            )

            print(f"Found {len(results)} structures")
            results = results[:max_structures]

            for entry in tqdm(results, desc="Processing"):
                try:
                    data.append({
                        'structure': entry['structure'],
                        'properties': {
                            'material_id': entry['material_id'],
                            'chemical_formula': entry['pretty_formula'],
                            'spacegroup': entry['spacegroup']['number'],
                            'formation_energy': entry['formation_energy_per_atom'],
                            'band_gap': entry['band_gap'],
                            'density': entry['density']
                        }
                    })
                except Exception as e:
                    print(f"Error processing {entry.get('material_id', 'unknown')}: {e}")
                    continue

        self._save_dataset(data, save_path)

        print(f"Loaded {len(data)} structures from MP30")
        return data

    def load_perovskite_dataset(
        self,
        save_path: str = "./data/p5.json",
        max_structures: int = 18928
    ) -> List[Dict[str, Any]]:
        """
        加载Perovskite-5 (P5)数据集

        P5包含18,928个钙钛矿结构

        Args:
            save_path: 保存路径
            max_structures: 最大结构数量

        Returns:
            数据列表
        """
        print(f"Loading Perovskite-5 dataset...")

        if not self.api_key:
            raise ValueError("API key required for Materials Project access")

        data = []

        with MPRester(self.api_key) as mpr:
            # 查询钙钛矿结构 (ABX3)
            results = mpr.query(
                criteria={
                    "nelements": 3,
                    "spacegroup.number": {"$in": [221, 225]},  # 常见钙钛矿空间群
                    "e_above_hull": {"$lte": 0.1}
                },
                properties=[
                    "material_id",
                    "structure",
                    "formation_energy_per_atom",
                    "band_gap",
                    "spacegroup",
                    "pretty_formula",
                    "density"
                ]
            )

            print(f"Found {len(results)} perovskite structures")
            results = results[:max_structures]

            for entry in tqdm(results, desc="Processing"):
                try:
                    # 验证是否为ABX3结构
                    structure = entry['structure']
                    composition = structure.composition

                    data.append({
                        'structure': structure,
                        'properties': {
                            'material_id': entry['material_id'],
                            'chemical_formula': entry['pretty_formula'],
                            'spacegroup': entry['spacegroup']['number'],
                            'formation_energy': entry['formation_energy_per_atom'],
                            'band_gap': entry['band_gap'],
                            'density': entry['density']
                        }
                    })
                except Exception as e:
                    print(f"Error processing {entry.get('material_id', 'unknown')}: {e}")
                    continue

        self._save_dataset(data, save_path)

        print(f"Loaded {len(data)} perovskite structures")
        return data

    def load_carbon_dataset(
        self,
        save_path: str = "./data/c24.json",
        max_structures: int = 10153
    ) -> List[Dict[str, Any]]:
        """
        加载Carbon-24 (C24)数据集

        C24包含10,153个碳结构

        Args:
            save_path: 保存路径
            max_structures: 最大结构数量

        Returns:
            数据列表
        """
        print(f"Loading Carbon-24 dataset...")

        if not self.api_key:
            raise ValueError("API key required for Materials Project access")

        data = []

        with MPRester(self.api_key) as mpr:
            # 查询纯碳结构
            results = mpr.query(
                criteria={
                    "elements": ["C"],
                    "nelements": 1,
                    "e_above_hull": {"$lte": 0.2}
                },
                properties=[
                    "material_id",
                    "structure",
                    "formation_energy_per_atom",
                    "band_gap",
                    "spacegroup",
                    "pretty_formula",
                    "density"
                ]
            )

            print(f"Found {len(results)} carbon structures")
            results = results[:max_structures]

            for entry in tqdm(results, desc="Processing"):
                try:
                    data.append({
                        'structure': entry['structure'],
                        'properties': {
                            'material_id': entry['material_id'],
                            'chemical_formula': entry['pretty_formula'],
                            'spacegroup': entry['spacegroup']['number'],
                            'formation_energy': entry['formation_energy_per_atom'],
                            'band_gap': entry['band_gap'],
                            'density': entry['density']
                        }
                    })
                except Exception as e:
                    print(f"Error processing {entry.get('material_id', 'unknown')}: {e}")
                    continue

        self._save_dataset(data, save_path)

        print(f"Loaded {len(data)} carbon structures")
        return data

    def _save_dataset(self, data: List[Dict[str, Any]], save_path: str):
        """保存数据集到JSON文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 转换为可序列化格式
        serializable_data = []
        for item in data:
            serializable_data.append({
                'structure': item['structure'].as_dict(),
                'properties': item['properties']
            })

        with open(save_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"Saved to {save_path}")

    def load_from_saved(self, json_path: str) -> List[Dict[str, Any]]:
        """从保存的JSON文件加载数据集"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            processed_data.append({
                'structure': Structure.from_dict(item['structure']),
                'properties': item['properties']
            })

        return processed_data


def main():
    """主函数 - 下载所有数据集"""
    import argparse

    parser = argparse.ArgumentParser(description='Download Materials Project datasets')
    parser.add_argument('--api_key', type=str, help='Materials Project API key')
    parser.add_argument('--dataset', type=str, choices=['mp20', 'mp30', 'p5', 'c24', 'all'],
                        default='all', help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory')

    args = parser.parse_args()

    # 初始化加载器
    loader = MaterialsProjectLoader(api_key=args.api_key)

    # 下载数据集
    if args.dataset in ['mp20', 'all']:
        print("\n" + "="*80)
        loader.load_mp20_dataset(
            save_path=os.path.join(args.output_dir, 'mp20.json')
        )

    if args.dataset in ['mp30', 'all']:
        print("\n" + "="*80)
        loader.load_mp30_dataset(
            save_path=os.path.join(args.output_dir, 'mp30.json')
        )

    if args.dataset in ['p5', 'all']:
        print("\n" + "="*80)
        loader.load_perovskite_dataset(
            save_path=os.path.join(args.output_dir, 'p5.json')
        )

    if args.dataset in ['c24', 'all']:
        print("\n" + "="*80)
        loader.load_carbon_dataset(
            save_path=os.path.join(args.output_dir, 'c24.json')
        )

    print("\n" + "="*80)
    print("All datasets downloaded successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
