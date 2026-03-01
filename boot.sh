#!/bin/bash
#
# Aegis Boot Script
# 启动 Aegis 应用 (前后端统一端口)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认配置 (will be overridden by aegis.yaml)
HOST="0.0.0.0"
PORT=8000
DEBUG=false
SKIP_FRONTEND=${SKIP_FRONTEND:-false}
DEV_MODE=${DEV_MODE:-false}

# 从 aegis.yaml 读取配置 (需要 Python + PyYAML)
read_config() {
    CONFIG_FILE="$SCRIPT_DIR/aegis.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        print_warn "aegis.yaml not found, using defaults."
        return
    fi

    # 使用 Python 解析 YAML （轻量、可靠）
    eval "$($PYTHON_CMD -c "
import yaml, sys
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f) or {}
s = cfg.get('server', {})
print(f\"YAML_HOST={s.get('host', '')}\")
print(f\"YAML_PORT={s.get('port', '')}\")
print(f\"YAML_DEBUG={str(s.get('debug', '')).lower()}\")
" 2>/dev/null)" || true

    [ -n "$YAML_HOST" ]  && HOST="$YAML_HOST"
    [ -n "$YAML_PORT" ]  && PORT="$YAML_PORT"
    [ -n "$YAML_DEBUG" ] && [ "$YAML_DEBUG" != "" ] && DEBUG="$YAML_DEBUG"
}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}"
    echo "    _    _____ ____ ___ ____  "
    echo "   / \  | ____/ ___|_ _/ ___| "
    echo "  / _ \ |  _|| |  _ | |\___ \ "
    echo " / ___ \| |__| |_| || | ___) |"
    echo "/_/   \_\_____\____|___|____/ "
    echo ""
    echo "  AI Image Generation Agent Framework"
    echo -e "${NC}"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Aegis Boot Script - 启动应用服务"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST       服务器主机 (default: $HOST)"
    echo "  -p, --port PORT       服务器端口 (default: $PORT)"
    echo "  -d, --debug           启用调试模式"
    echo "  --dev                 开发模式 (自动重载)"
    echo "  --skip-frontend       跳过前端构建"
    echo "  --help                显示帮助信息"
    echo ""
    echo "Configuration:"
    echo "  Settings are read from aegis.yaml in the project root."
    echo "  Command-line flags override aegis.yaml values."
    echo ""
    echo "Examples:"
    echo "  $0                    # 默认启动"
    echo "  $0 -p 3000            # 指定端口"
    echo "  $0 --dev              # 开发模式"
    echo "  $0 --skip-frontend    # 仅启动后端"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            DEBUG=true
            shift
            ;;
        --skip-frontend)
            SKIP_FRONTEND=true
            shift
            ;;
        --help)
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

# 检查 Python
check_python() {
    # 尝试激活conda环境
    if [ -f "$HOME/miniconda3/bin/activate" ]; then
        source "$HOME/miniconda3/bin/activate"
        if conda activate aegis 2>/dev/null; then
            PYTHON_CMD=python
            PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
            print_info "Python: $PYTHON_VERSION (conda env: aegis)"
            return
        fi
    fi
    
    # 如果没有conda，尝试 python3，然后是 python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        print_error "Python not found. Please install Python 3.12+"
        print_error "Option 1: conda create -n aegis python=3.12"
        print_error "Option 2: sudo apt install python3 python3-pip python3-venv"
        exit 1
    fi
    
    # 检查 Python 版本
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_info "Python: $PYTHON_VERSION (using $PYTHON_CMD)"
}

# 检查 Node.js
check_node() {
    if ! command -v node &> /dev/null; then
        print_warn "Node.js not found. Frontend build will be skipped."
        SKIP_FRONTEND=true
        return
    fi
    print_info "Node.js: $(node --version)"
}

# 设置 Python 虚拟环境
setup_venv() {
    # 如果已经在conda环境中，跳过venv设置
    if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" = "aegis" ]; then
        print_info "Using conda environment: aegis"
        return
    fi
    
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -d "venv" ]; then
            print_info "Activating virtual environment..."
            source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || {
                print_warn "Failed to activate venv. Using system Python."
            }
        elif [ -d "backend/venv" ]; then
            print_info "Activating virtual environment..."
            source backend/venv/bin/activate 2>/dev/null || source backend/venv/Scripts/activate 2>/dev/null || {
                print_warn "Failed to activate venv. Using system Python."
            }
        else
            print_warn "No virtual environment found."
            print_info "Creating virtual environment..."
            $PYTHON_CMD -m venv venv || {
                print_error "Failed to create virtual environment."
                print_error "Please install: sudo apt install python3-venv"
                exit 1
            }
            source venv/bin/activate
            print_info "Virtual environment created and activated."
        fi
    else
        print_info "Already in virtual environment: $VIRTUAL_ENV"
    fi
}

# 安装后端依赖
install_backend_deps() {
    print_info "Checking backend dependencies..."
    
    # 确保 pip 可用
    if ! command -v pip &> /dev/null; then
        print_info "Installing pip..."
        $PYTHON_CMD -m ensurepip --upgrade 2>/dev/null || {
            print_error "pip not found. Please install: sudo apt install python3-pip"
            exit 1
        }
    fi
    
    if [ -f "backend/requirements.txt" ]; then
        # 先升级 pip
        pip install --upgrade pip -q 2>/dev/null || true
        
        # 安装依赖
        pip install -q -r backend/requirements.txt 2>/dev/null || {
            print_warn "Some dependencies may be missing. Installing..."
            pip install -r backend/requirements.txt
        }
    else
        print_warn "requirements.txt not found at backend/requirements.txt"
    fi
}

# 构建前端
build_frontend() {
    if [ "$SKIP_FRONTEND" = true ]; then
        print_info "Skipping frontend build."
        return
    fi
    
    if [ ! -d "frontend" ]; then
        print_warn "Frontend directory not found. Skipping build."
        return
    fi
    
    print_info "Building frontend..."
    cd frontend
    
    # 安装依赖
    if [ ! -d "node_modules" ]; then
        print_info "Installing frontend dependencies..."
        npm install
    fi
    
    # 构建
    npm run build
    
    cd ..
    print_info "Frontend build completed."
}

# 创建数据目录
setup_directories() {
    mkdir -p data
    mkdir -p logs
}

# 启动服务
start_server() {
    print_info "Starting Aegis server..."

    # ── 端口可用性预检 ──────────────────────────
    # WSL2 环境中 Windows winnat 可能静默保留某些端口段 (如 8000-8080),
    # Linux 工具 (ss/lsof) 看不到，但 bind() 会失败。
    # 在启动 uvicorn 前先用 Python 快速检测。
    if ! $PYTHON_CMD -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('$HOST', $PORT))
    s.close()
except OSError:
    print(f'ERROR: Port $PORT is unavailable (likely reserved by Windows/Hyper-V winnat).')
    print(f'Fix options:')
    print(f'  1. Change port in aegis.yaml to 3000/5000/9000 etc.')
    print(f'  2. On Windows PowerShell (Admin): net stop winnat && net start winnat')
    print(f'  3. Reserve port: netsh int ipv4 add excludedportrange protocol=tcp startport=$PORT numberofports=1')
    sys.exit(1)
" 2>/dev/null; then
        print_error "Port $PORT is not available. See above for fix options."
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo "  Server: http://${HOST}:${PORT}"
    echo "  API Docs: http://${HOST}:${PORT}/docs"
    echo "  Debug: $DEBUG"
    echo "=========================================="
    echo ""
    
    # Set AEGIS_CONFIG so the Python backend finds the config file
    export AEGIS_CONFIG="$SCRIPT_DIR/aegis.yaml"
    
    cd backend
    
    if [ "$DEV_MODE" = true ]; then
        print_info "Starting in development mode with auto-reload..."
        $PYTHON_CMD -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
    else
        $PYTHON_CMD -m uvicorn app.main:app --host "$HOST" --port "$PORT"
    fi
}

# 清理函数
cleanup() {
    print_info "Shutting down..."
    exit 0
}

trap cleanup SIGINT SIGTERM

# 主函数
main() {
    print_header
    
    check_python
    setup_venv
    read_config
    check_node
    install_backend_deps
    setup_directories
    build_frontend
    start_server
}

main
