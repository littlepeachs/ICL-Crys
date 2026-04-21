"""
Condition-Structure Aware Hybrid Crystal Instruction Tuning
构建晶体生成的指令微调数据集
"""

import random
from typing import List, Dict, Any, Optional
from pymatgen.core import Structure
from crystal_tokenization import CrystalTokenizer


class InstructionBuilder:
    """晶体指令构建器"""

    def __init__(self, tokenizer: CrystalTokenizer):
        self.tokenizer = tokenizer

    def build_zero_shot_instruction(self, property_name: str, property_desc: str) -> str:
        """构建零样本指令"""
        instruction = f"""### Instruction: Below is a description of a bulk material. [Condition Description]. Generate the space group symbol, a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice:
### Response: [Crystal String]."""

        instruction = instruction.replace(
            "[Condition Description]",
            f"The {property_name} is {property_desc}"
        )

        return instruction

    def build_few_shot_instruction(
        self,
        examples: List[Dict[str, Any]],
        target_property: Dict[str, Any],
        k_shot: int = 3
    ) -> str:
        """
        构建少样本指令

        Args:
            examples: 示例晶体列表，每个包含 {'structure': Structure, 'properties': dict}
            target_property: 目标属性字典
            k_shot: 少样本数量

        Returns:
            构建好的指令文本
        """
        # 选择k个示例
        selected_examples = examples[:k_shot]

        instruction = "### Instruction: Below is three description of bulk materials.\n"

        # 添加示例
        for i, example in enumerate(selected_examples, 1):
            structure = example['structure']
            properties = example['properties']

            instruction += f"### First Example:\n" if i == 1 else f"### {'Second' if i == 2 else 'Third'} Example:\n"

            # 添加属性描述
            prop_desc = self._format_properties(properties)
            instruction += f"### {prop_desc}\n"

            # 添加晶体结构
            crystal_text = self.tokenizer.tokenize(structure, use_sgs=True)
            instruction += f"### {crystal_text}\n"

        # 添加目标查询
        target_desc = self._format_properties(target_property)
        instruction += f"### {target_desc}. Based on the three examples provided, generate the space group symbol, a description of the lengths and angles of the lattice vectors, along with the element type and coordinates for each atom within the lattice:\n"
        instruction += "### Response: [Crystal String]."

        return instruction

    def build_property_prediction_instruction(
        self,
        structure: Structure,
        property_name: str,
        mask_token: str = "[MASK]"
    ) -> str:
        """
        构建属性预测指令（多任务学习）

        Args:
            structure: 晶体结构
            property_name: 要预测的属性名称
            mask_token: 掩码token

        Returns:
            属性预测指令
        """
        crystal_text = self.tokenizer.tokenize(structure, use_sgs=True)

        instruction = f"""### Instruction: Below is a partial description of a bulk material where the [{property_name}] has been replaced with the string "{mask_token}":
### The [{property_name}] is {mask_token}.
### [Crys Str]
### Generate the [{property_name}] that could replace {mask_token} in the bulk material:
### Response: [Property Value]."""

        instruction = instruction.replace("[Crys Str]", crystal_text)

        return instruction

    def _format_properties(self, properties: Dict[str, Any]) -> str:
        """格式化属性描述"""
        prop_strs = []

        if 'chemical_formula' in properties:
            prop_strs.append(f"The chemical formula is {properties['chemical_formula']}")

        if 'spacegroup' in properties:
            prop_strs.append(f"The spacegroup number is {properties['spacegroup']}")

        if 'formation_energy' in properties:
            prop_strs.append(f"The formation energy per atom is {properties['formation_energy']:.4f}")

        if 'band_gap' in properties:
            prop_strs.append(f"The band gap is {properties['band_gap']:.4f}")

        return ". ".join(prop_strs)

    def build_conditional_generation_instruction(
        self,
        properties: Dict[str, Any],
        use_few_shot: bool = False,
        examples: Optional[List[Dict[str, Any]]] = None,
        k_shot: int = 3
    ) -> str:
        """
        构建条件生成指令

        Args:
            properties: 目标属性
            use_few_shot: 是否使用少样本
            examples: 示例数据
            k_shot: 少样本数量

        Returns:
            条件生成指令
        """
        if use_few_shot and examples:
            return self.build_few_shot_instruction(examples, properties, k_shot)
        else:
            prop_desc = self._format_properties(properties)
            return self.build_zero_shot_instruction("properties", prop_desc)

    def build_unconditional_generation_instruction(self) -> str:
        """构建无条件生成指令"""
        instruction = """### Instruction: Below is a description of a bulk material. Generate the space group symbol, a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice:
### Response: [Crystal String]."""

        return instruction


def test_instruction_builder():
    """测试指令构建器"""
    from pymatgen.core import Lattice, Structure

    # 创建tokenizer
    tokenizer = CrystalTokenizer()
    builder = InstructionBuilder(tokenizer)

    # 创建示例结构
    lattice = Lattice.cubic(5.64)
    structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # 测试零样本指令
    print("Zero-shot Instruction:")
    zero_shot = builder.build_zero_shot_instruction(
        "formation energy",
        "-0.5 eV/atom"
    )
    print(zero_shot)
    print("\n" + "="*80 + "\n")

    # 测试少样本指令
    print("Few-shot Instruction:")
    examples = [
        {
            'structure': structure,
            'properties': {
                'chemical_formula': 'NaCl',
                'spacegroup': 225,
                'formation_energy': -0.5,
                'band_gap': 5.0
            }
        }
    ] * 3

    target_properties = {
        'chemical_formula': 'MgO',
        'spacegroup': 225,
        'formation_energy': -0.6,
        'band_gap': 7.8
    }

    few_shot = builder.build_few_shot_instruction(examples, target_properties, k_shot=3)
    print(few_shot)
    print("\n" + "="*80 + "\n")

    # 测试属性预测指令
    print("Property Prediction Instruction:")
    prop_pred = builder.build_property_prediction_instruction(
        structure,
        "formation energy"
    )
    print(prop_pred)


if __name__ == "__main__":
    test_instruction_builder()
