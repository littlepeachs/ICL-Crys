#!/bin/bash

# CrystalICL Quick Start Script
# 快速启动脚本

echo "=========================================="
echo "CrystalICL Quick Start"
echo "=========================================="

# 检查Python版本
echo ""
echo "Checking Python version..."
python --version

# 安装依赖
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# 运行测试
echo ""
echo "Running module tests..."
python test_modules.py

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "All tests passed! Starting training..."
    echo "=========================================="

    # 运行训练（使用示例数据）
    python run_crystalicl.py \
        --use_sample_data \
        --num_samples 50 \
        --do_train \
        --do_eval \
        --num_epochs 1 \
        --batch_size 1 \
        --learning_rate 5e-4 \
        --use_few_shot \
        --k_shot 3 \
        --output_dir ./crystalicl_output

    echo ""
    echo "=========================================="
    echo "Training completed!"
    echo "Model saved to: ./crystalicl_output"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Tests failed! Please check the errors above."
    echo "=========================================="
    exit 1
fi
