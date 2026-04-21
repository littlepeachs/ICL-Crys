"""
CrystalICL Training Script with Qwen3-8B
使用Qwen3-8B模型训练CrystalICL
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

from crystal_tokenization import CrystalTokenizer
from instruction_builder import InstructionBuilder
from example_selector import ExampleSelector


class CrystalInstructionDataset(Dataset):
    """晶体指令数据集"""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        crystal_tokenizer: CrystalTokenizer,
        instruction_builder: InstructionBuilder,
        example_selector: ExampleSelector,
        max_length: int = 2048,
        use_few_shot: bool = True,
        k_shot: int = 3,
        include_property_prediction: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.crystal_tokenizer = crystal_tokenizer
        self.instruction_builder = instruction_builder
        self.example_selector = example_selector
        self.max_length = max_length
        self.use_few_shot = use_few_shot
        self.k_shot = k_shot
        self.include_property_prediction = include_property_prediction

        # 构建指令数据
        self.instructions = self._build_instructions()

    def _build_instructions(self) -> List[Dict[str, str]]:
        """构建指令数据"""
        instructions = []

        for i, sample in enumerate(tqdm(self.data, desc="Building instructions")):
            structure = sample['structure']
            properties = sample.get('properties', {})

            # 1. 条件生成指令
            if self.use_few_shot and len(self.data) > self.k_shot:
                # 选择示例（排除当前样本）
                other_samples = [s for j, s in enumerate(self.data) if j != i]
                examples = self.example_selector.condition_structure_based_selection(
                    other_samples,
                    properties,
                    structure,
                    k=self.k_shot
                )
                instruction = self.instruction_builder.build_few_shot_instruction(
                    examples, properties, self.k_shot
                )
            else:
                instruction = self.instruction_builder.build_conditional_generation_instruction(
                    properties, use_few_shot=False
                )

            # 生成响应（晶体结构文本）
            response = self.crystal_tokenizer.tokenize(structure, use_sgs=True)

            instructions.append({
                'instruction': instruction,
                'response': response,
                'type': 'conditional_generation'
            })

            # 2. 属性预测指令（多任务学习）
            if self.include_property_prediction:
                for prop_name in ['formation_energy', 'band_gap']:
                    if prop_name in properties:
                        prop_instruction = self.instruction_builder.build_property_prediction_instruction(
                            structure, prop_name
                        )
                        prop_value = str(properties[prop_name])

                        instructions.append({
                            'instruction': prop_instruction,
                            'response': prop_value,
                            'type': 'property_prediction'
                        })

        return instructions

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        item = self.instructions[idx]

        # 构建完整的输入文本
        full_text = f"{item['instruction']}\n{item['response']}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # 准备标签（只计算response部分的损失）
        instruction_encoding = self.tokenizer(
            item['instruction'],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )

        labels = encoding['input_ids'].clone()
        # 将instruction部分的标签设为-100（不计算损失）
        instruction_length = len(instruction_encoding['input_ids'])
        labels[0, :instruction_length] = -100

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


class CrystalICLTrainer:
    """CrystalICL训练器"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        output_dir: str = "./crystalicl_qwen3_8b",
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # 初始化组件
        self.crystal_tokenizer = CrystalTokenizer()
        self.instruction_builder = InstructionBuilder(self.crystal_tokenizer)
        self.example_selector = ExampleSelector()

        # 加载模型和tokenizer
        self._load_model()

    def _load_model(self):
        """加载模型和tokenizer"""
        print(f"Loading model: {self.model_name}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='right'
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # 应用LoRA
        if self.use_lora:
            print("Applying LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: List[Dict[str, Any]] = None,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 5e-4,
        use_few_shot: bool = True,
        k_shot: int = 3
    ):
        """
        训练模型

        Args:
            train_data: 训练数据
            eval_data: 验证数据
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            use_few_shot: 是否使用少样本
            k_shot: 少样本数量
        """
        print("Preparing datasets...")

        # 创建数据集
        train_dataset = CrystalInstructionDataset(
            train_data,
            self.tokenizer,
            self.crystal_tokenizer,
            self.instruction_builder,
            self.example_selector,
            use_few_shot=use_few_shot,
            k_shot=k_shot
        )

        eval_dataset = None
        if eval_data:
            eval_dataset = CrystalInstructionDataset(
                eval_data,
                self.tokenizer,
                self.crystal_tokenizer,
                self.instruction_builder,
                self.example_selector,
                use_few_shot=use_few_shot,
                k_shot=k_shot
            )

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500 if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            fp16=False,
            bf16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=4,
        )

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # 开始训练
        print("Starting training...")
        trainer.train()

        # 保存模型
        print(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def generate(
        self,
        instruction: str,
        max_new_tokens: int = 512,
        temperature: float = 0.9,
        top_p: float = 0.9
    ) -> str:
        """
        生成晶体结构

        Args:
            instruction: 输入指令
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数

        Returns:
            生成的文本
        """
        self.model.eval()

        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取response部分
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text

        return response


def main():
    """主函数"""
    # 示例：创建训练数据
    from pymatgen.core import Lattice, Structure

    print("Creating sample training data...")
    train_data = []

    # 创建一些示例结构
    structures_info = [
        {
            'lattice': Lattice.cubic(5.64),
            'species': ['Na', 'Cl'],
            'coords': [[0, 0, 0], [0.5, 0.5, 0.5]],
            'properties': {
                'chemical_formula': 'NaCl',
                'spacegroup': 225,
                'formation_energy': -0.5,
                'band_gap': 5.0
            }
        },
        {
            'lattice': Lattice.cubic(4.21),
            'species': ['Mg', 'O'],
            'coords': [[0, 0, 0], [0.5, 0.5, 0.5]],
            'properties': {
                'chemical_formula': 'MgO',
                'spacegroup': 225,
                'formation_energy': -0.6,
                'band_gap': 7.8
            }
        }
    ]

    for info in structures_info:
        structure = Structure(info['lattice'], info['species'], info['coords'])
        train_data.append({
            'structure': structure,
            'properties': info['properties']
        })

    # 初始化训练器
    trainer = CrystalICLTrainer(
        model_name="Qwen/Qwen3-8B",  # 使用Qwen3-8B
        output_dir="./crystalicl_qwen3_8b_output",
        use_lora=True,
        lora_rank=8,
        lora_alpha=32
    )

    # 训练模型
    trainer.train(
        train_data=train_data,
        num_epochs=3,
        batch_size=1,
        learning_rate=5e-4,
        use_few_shot=True,
        k_shot=3
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
