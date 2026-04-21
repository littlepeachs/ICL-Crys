"""
Main script to run CrystalICL training and evaluation
完整的训练和评估流程
"""

import argparse
import os
import json
from data_loader import CrystalDataLoader
from train_crystalicl import CrystalICLTrainer
from evaluate import CrystalEvaluator


def prepare_data(args):
    """准备数据"""
    print("="*80)
    print("Step 1: Preparing Data")
    print("="*80)

    loader = CrystalDataLoader(data_dir=args.data_dir)

    if args.use_sample_data:
        # 使用示例数据
        print(f"Creating sample dataset with {args.num_samples} samples...")
        data = loader.create_sample_dataset(num_samples=args.num_samples)
    else:
        # 从文件加载数据
        if args.data_format == 'json':
            print(f"Loading data from {args.data_path}...")
            data = loader.load_from_json(args.data_path)
        elif args.data_format == 'cif':
            print(f"Loading CIF files from {args.data_path}...")
            data = loader.load_from_cif_dir(
                args.data_path,
                properties_file=args.properties_file
            )
        else:
            raise ValueError(f"Unsupported data format: {args.data_format}")

    print(f"Total samples: {len(data)}")

    # 划分数据集
    print("\nSplitting dataset...")
    splits = loader.split_dataset(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    print(f"Train: {len(splits['train'])} samples")
    print(f"Val: {len(splits['val'])} samples")
    print(f"Test: {len(splits['test'])} samples")

    # 保存数据
    os.makedirs(args.data_dir, exist_ok=True)
    loader.save_to_json(splits['train'], os.path.join(args.data_dir, 'train.json'))
    loader.save_to_json(splits['val'], os.path.join(args.data_dir, 'val.json'))
    loader.save_to_json(splits['test'], os.path.join(args.data_dir, 'test.json'))

    return splits


def train_model(args, train_data, val_data):
    """训练模型"""
    print("\n" + "="*80)
    print("Step 2: Training Model")
    print("="*80)

    trainer = CrystalICLTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    trainer.train(
        train_data=train_data,
        eval_data=val_data if args.do_eval else None,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_few_shot=args.use_few_shot,
        k_shot=args.k_shot
    )

    print(f"\nModel saved to {args.output_dir}")


def evaluate_model(args, test_data):
    """评估模型"""
    print("\n" + "="*80)
    print("Step 3: Evaluating Model")
    print("="*80)

    evaluator = CrystalEvaluator(model_path=args.output_dir)

    results = evaluator.evaluate_model(
        test_data=test_data,
        num_samples=args.eval_samples
    )

    # 保存结果
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to {results_path}")

    # 打印摘要
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description='CrystalICL Training and Evaluation')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file or directory')
    parser.add_argument('--data_format', type=str, default='json',
                        choices=['json', 'cif'],
                        help='Data format')
    parser.add_argument('--properties_file', type=str, default=None,
                        help='Properties file for CIF data')
    parser.add_argument('--use_sample_data', action='store_true',
                        help='Use sample data for testing')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for sample data')

    # 数据划分参数
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B',
                        help='Base model name')
    parser.add_argument('--output_dir', type=str, default='./crystalicl_qwen3_8b_output',
                        help='Output directory')

    # LoRA参数
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Use LoRA for training')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--use_few_shot', action='store_true', default=True,
                        help='Use few-shot learning')
    parser.add_argument('--k_shot', type=int, default=3,
                        help='Number of shots for few-shot learning')

    # 运行模式
    parser.add_argument('--do_train', action='store_true',
                        help='Run training')
    parser.add_argument('--do_eval', action='store_true',
                        help='Run evaluation')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='Number of samples for evaluation')

    args = parser.parse_args()

    # 准备数据
    if args.use_sample_data or args.data_path:
        splits = prepare_data(args)
    else:
        # 从已保存的数据加载
        loader = CrystalDataLoader(data_dir=args.data_dir)
        splits = {
            'train': loader.load_from_json(os.path.join(args.data_dir, 'train.json')),
            'val': loader.load_from_json(os.path.join(args.data_dir, 'val.json')),
            'test': loader.load_from_json(os.path.join(args.data_dir, 'test.json'))
        }

    # 训练
    if args.do_train:
        train_model(args, splits['train'], splits['val'])

    # 评估
    if args.do_eval:
        evaluate_model(args, splits['test'])

    print("\n" + "="*80)
    print("All tasks completed!")
    print("="*80)


if __name__ == "__main__":
    main()
