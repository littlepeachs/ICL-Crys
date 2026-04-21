"""
Data Loader for Crystal Generation Datasets
加载MP20, MP30, P5, C24等数据集
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
import numpy as np


class CrystalDataLoader:
    """晶体数据加载器"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        从JSON文件加载数据

        JSON格式示例:
        [
            {
                "structure": {...},  # pymatgen Structure dict
                "properties": {
                    "formation_energy": -0.5,
                    "band_gap": 5.0,
                    "spacegroup": 225,
                    "chemical_formula": "NaCl"
                }
            }
        ]
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            structure = Structure.from_dict(item['structure'])
            properties = item.get('properties', {})

            processed_data.append({
                'structure': structure,
                'properties': properties
            })

        return processed_data

    def load_from_cif_dir(
        self,
        cif_dir: str,
        properties_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        从CIF文件目录加载数据

        Args:
            cif_dir: CIF文件目录
            properties_file: 属性文件路径（JSON格式）

        Returns:
            加载的数据列表
        """
        # 加载属性数据
        properties_dict = {}
        if properties_file and os.path.exists(properties_file):
            with open(properties_file, 'r') as f:
                properties_dict = json.load(f)

        processed_data = []

        # 遍历CIF文件
        for filename in os.listdir(cif_dir):
            if not filename.endswith('.cif'):
                continue

            cif_path = os.path.join(cif_dir, filename)
            material_id = filename.replace('.cif', '')

            try:
                structure = Structure.from_file(cif_path)

                # 获取属性
                properties = properties_dict.get(material_id, {})

                # 自动计算一些基本属性
                if 'chemical_formula' not in properties:
                    properties['chemical_formula'] = structure.composition.reduced_formula

                processed_data.append({
                    'structure': structure,
                    'properties': properties,
                    'material_id': material_id
                })

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

        return processed_data

    def create_sample_dataset(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        创建示例数据集（用于测试）

        Args:
            num_samples: 样本数量

        Returns:
            示例数据集
        """
        from pymatgen.core import Lattice
        import random

        data = []

        # 定义一些常见的晶体结构模板
        templates = [
            {
                'name': 'rocksalt',
                'species': ['Na', 'Cl'],
                'coords': [[0, 0, 0], [0.5, 0.5, 0.5]],
                'lattice_range': (5.0, 6.0),
                'spacegroup': 225
            },
            {
                'name': 'perovskite',
                'species': ['Ba', 'Ti', 'O', 'O', 'O'],
                'coords': [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
                'lattice_range': (3.8, 4.2),
                'spacegroup': 221
            },
            {
                'name': 'zincblende',
                'species': ['Zn', 'S'],
                'coords': [[0, 0, 0], [0.25, 0.25, 0.25]],
                'lattice_range': (5.2, 5.8),
                'spacegroup': 216
            }
        ]

        for i in range(num_samples):
            template = random.choice(templates)

            # 随机晶格参数
            a = random.uniform(*template['lattice_range'])
            lattice = Lattice.cubic(a)

            structure = Structure(
                lattice,
                template['species'],
                template['coords']
            )

            # 生成随机属性
            properties = {
                'chemical_formula': structure.composition.reduced_formula,
                'spacegroup': template['spacegroup'],
                'formation_energy': random.uniform(-2.0, 0.0),
                'band_gap': random.uniform(0.0, 10.0)
            }

            data.append({
                'structure': structure,
                'properties': properties,
                'template': template['name']
            })

        return data

    def save_to_json(self, data: List[Dict[str, Any]], output_path: str):
        """保存数据到JSON文件"""
        serializable_data = []

        for item in data:
            serializable_item = {
                'structure': item['structure'].as_dict(),
                'properties': item['properties']
            }

            # 添加其他字段
            for key in item:
                if key not in ['structure', 'properties']:
                    serializable_item[key] = item[key]

            serializable_data.append(serializable_item)

        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"Saved {len(data)} samples to {output_path}")

    def split_dataset(
        self,
        data: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        划分数据集

        Args:
            data: 完整数据集
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子

        Returns:
            包含train, val, test的字典
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        np.random.seed(random_seed)
        indices = np.random.permutation(len(data))

        n_train = int(len(data) * train_ratio)
        n_val = int(len(data) * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        return {
            'train': [data[i] for i in train_indices],
            'val': [data[i] for i in val_indices],
            'test': [data[i] for i in test_indices]
        }


def main():
    """测试数据加载器"""
    loader = CrystalDataLoader(data_dir="./data")

    # 创建示例数据集
    print("Creating sample dataset...")
    data = loader.create_sample_dataset(num_samples=100)

    print(f"Created {len(data)} samples")
    print(f"First sample: {data[0]['properties']}")

    # 划分数据集
    print("\nSplitting dataset...")
    splits = loader.split_dataset(data)

    print(f"Train: {len(splits['train'])} samples")
    print(f"Val: {len(splits['val'])} samples")
    print(f"Test: {len(splits['test'])} samples")

    # 保存数据
    print("\nSaving datasets...")
    loader.save_to_json(splits['train'], "./data/train.json")
    loader.save_to_json(splits['val'], "./data/val.json")
    loader.save_to_json(splits['test'], "./data/test.json")

    print("Done!")


if __name__ == "__main__":
    main()
