#!/bin/bash
#
# Aegis Boot Script
# 启动 Aegis 应用 (前后端统一端口)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认配置
HOST=${AEGIS_HOST:-"0.0.0.0"}
PORT=${AEGIS_PORT:-8000}
DEBUG=${AEGIS_DEBUG:-false}
SKIP_FRONTEND=${SKIP_FRONTEND:-false}
DEV_MODE=${DEV_MODE:-false}

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
    echo "Environment Variables:"
    echo "  AEGIS_HOST            服务器主机"
    echo "  AEGIS_PORT            服务器端口"
    echo "  AEGIS_DEBUG           调试模式"
    echo "  AEGIS_DATABASE_URL    数据库连接URL"
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
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.12+"
        exit 1
    fi
    print_info "Python: $(python --version)"
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
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -d "venv" ]; then
            print_info "Activating virtual environment..."
            source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true
        elif [ -d "backend/venv" ]; then
            print_info "Activating virtual environment..."
            source backend/venv/bin/activate 2>/dev/null || source backend/venv/Scripts/activate 2>/dev/null || true
        else
            print_warn "No virtual environment found. Using system Python."
        fi
    fi
}

# 安装后端依赖
install_backend_deps() {
    print_info "Checking backend dependencies..."
    
    if [ -f "backend/requirements.txt" ]; then
        pip install -q -r backend/requirements.txt 2>/dev/null || {
            print_warn "Some dependencies may be missing. Installing..."
            pip install -r backend/requirements.txt
        }
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
    echo ""
    echo "=========================================="
    echo "  Server: http://${HOST}:${PORT}"
    echo "  API Docs: http://${HOST}:${PORT}/docs"
    echo "  Debug: $DEBUG"
    echo "=========================================="
    echo ""
    
    export AEGIS_HOST=$HOST
    export AEGIS_PORT=$PORT
    export AEGIS_DEBUG=$DEBUG
    
    cd backend
    
    if [ "$DEV_MODE" = true ]; then
        print_info "Starting in development mode with auto-reload..."
        python -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
    else
        python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
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
    check_node
    setup_venv
    install_backend_deps
    setup_directories
    build_frontend
    start_server
}

main
