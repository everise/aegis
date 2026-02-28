#!/bin/bash
#
# Aegis RL Training Script
# 启动强化学习训练流程
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认配置
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
BUFFER_SIZE=${BUFFER_SIZE:-10000}
DISCOUNT_FACTOR=${DISCOUNT_FACTOR:-0.99}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
LOG_DIR=${LOG_DIR:-"./logs"}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Aegis RL Training Script - 启动强化学习训练"
    echo ""
    echo "Options:"
    echo "  -e, --epochs NUM        训练轮数 (default: $EPOCHS)"
    echo "  -b, --batch-size NUM    批次大小 (default: $BATCH_SIZE)"
    echo "  -l, --learning-rate NUM 学习率 (default: $LEARNING_RATE)"
    echo "  -r, --buffer-size NUM   回放缓冲区大小 (default: $BUFFER_SIZE)"
    echo "  -g, --gamma NUM         折扣因子 (default: $DISCOUNT_FACTOR)"
    echo "  -c, --checkpoint DIR    检查点目录 (default: $CHECKPOINT_DIR)"
    echo "  --log-dir DIR           日志目录 (default: $LOG_DIR)"
    echo "  --resume PATH           从检查点恢复训练"
    echo "  --eval-only             仅评估模式"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  $0                              # 使用默认配置训练"
    echo "  $0 -e 200 -b 64                 # 训练200轮，批次大小64"
    echo "  $0 --resume checkpoints/latest  # 从检查点恢复"
    echo ""
}

# 解析命令行参数
RESUME_PATH=""
EVAL_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -r|--buffer-size)
            BUFFER_SIZE="$2"
            shift 2
            ;;
        -g|--gamma)
            DISCOUNT_FACTOR="$2"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# 检查 Python 环境
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.12+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    print_info "Python version: $PYTHON_VERSION"
}

# 检查依赖
check_dependencies() {
    print_info "Checking dependencies..."
    
    if [ ! -f "backend/requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # 检查虚拟环境
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warn "No virtual environment detected. Consider using one."
        
        if [ -d "venv" ]; then
            print_info "Found venv directory. Activating..."
            source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true
        fi
    fi
}

# 创建必要目录
setup_directories() {
    print_info "Setting up directories..."
    
    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "data"
}

# 训练配置
print_config() {
    echo ""
    echo "=========================================="
    echo "         Aegis RL Training Config        "
    echo "=========================================="
    echo "  Epochs:          $EPOCHS"
    echo "  Batch Size:      $BATCH_SIZE"
    echo "  Learning Rate:   $LEARNING_RATE"
    echo "  Buffer Size:     $BUFFER_SIZE"
    echo "  Discount Factor: $DISCOUNT_FACTOR"
    echo "  Checkpoint Dir:  $CHECKPOINT_DIR"
    echo "  Log Dir:         $LOG_DIR"
    if [ -n "$RESUME_PATH" ]; then
        echo "  Resume From:     $RESUME_PATH"
    fi
    if [ "$EVAL_ONLY" = true ]; then
        echo "  Mode:            Evaluation Only"
    fi
    echo "=========================================="
    echo ""
}

# 运行训练
run_training() {
    print_info "Starting RL training..."
    
    # Point backend to the project-root config file
    export AEGIS_CONFIG="$SCRIPT_DIR/aegis.yaml"
    
    TRAIN_ARGS=""
    
    if [ -n "$RESUME_PATH" ]; then
        TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME_PATH"
    fi
    
    if [ "$EVAL_ONLY" = true ]; then
        TRAIN_ARGS="$TRAIN_ARGS --eval-only"
    fi
    
    cd backend
    
    python -c "
import asyncio
import sys
sys.path.insert(0, '..')

from rl.trainer import RLTrainer, TrainingConfig
from rl.trajectory import TrajectoryBuilder

async def main():
    config = TrainingConfig(
        epochs=$EPOCHS,
        batch_size=$BATCH_SIZE,
        learning_rate=$LEARNING_RATE,
        checkpoint_dir='$CHECKPOINT_DIR',
    )
    
    trainer = RLTrainer(config=config)
    
    print('Initializing trainer...')
    print(f'Config: {config.to_dict()}')
    
    # 训练循环 (示例)
    print('Training started. Press Ctrl+C to stop.')
    
    try:
        async for metrics in trainer.train($EPOCHS):
            print(f'Epoch {metrics.epoch}/{$EPOCHS} - Loss: {metrics.loss:.4f}, Reward: {metrics.avg_reward:.4f}')
    except KeyboardInterrupt:
        print('\\nTraining interrupted by user.')
    
    print('Training completed.')
    summary = trainer.get_summary()
    print(f'Summary: {summary}')

asyncio.run(main())
"
    
    cd ..
}

# 主函数
main() {
    print_info "Aegis RL Training Script"
    print_info "========================"
    
    check_python
    check_dependencies
    setup_directories
    print_config
    
    run_training
    
    print_info "Training script completed."
}

main
