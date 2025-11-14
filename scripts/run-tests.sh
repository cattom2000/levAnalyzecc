#!/bin/bash

# 测试运行脚本
# 支持多种测试模式和配置

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
PYTHON_VERSION="3.11"
TEST_TYPE="all"
COVERAGE=true
PARALLEL=true
VERBOSE=true
DOCKER=false
HTML_REPORT=false

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项] [测试类型]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -v, --version       指定Python版本 (默认: 3.11)"
    echo "  -c, --coverage      启用覆盖率报告 (默认: true)"
    echo "  -p, --parallel      并行运行测试 (默认: true)"
    echo "  -d, --docker        使用Docker运行测试"
    echo "  -r, --html-report   生成HTML报告"
    echo "  -q, --quiet         静默模式"
    echo ""
    echo "测试类型:"
    echo "  all                 运行所有测试 (默认)"
    echo "  unit                单元测试"
    echo "  integration         集成测试"
    echo "  data-quality        数据质量测试"
    echo "  performance         性能测试"
    echo "  quick               快速检查"
    echo ""
    echo "示例:"
    echo "  $0                  # 运行所有测试"
    echo "  $0 unit             # 只运行单元测试"
    echo "  $0 -d all           # 使用Docker运行所有测试"
    echo "  $0 -r -p integration # 并行运行集成测试并生成HTML报告"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        --no-parallel)
            PARALLEL=false
            shift
            ;;
        -d|--docker)
            DOCKER=true
            shift
            ;;
        -r|--html-report)
            HTML_REPORT=true
            shift
            ;;
        -q|--quiet)
            VERBOSE=false
            shift
            ;;
        unit|integration|data-quality|performance|quick|all)
            TEST_TYPE="$1"
            shift
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 设置环境变量
export TESTING=true
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 构建pytest命令
build_pytest_command() {
    local cmd="pytest"

    # 添加测试路径
    case $TEST_TYPE in
        unit)
            cmd="$cmd tests/ -m unit"
            ;;
        integration)
            cmd="$cmd tests/ -m integration"
            ;;
        data-quality)
            cmd="$cmd tests/ -m data_quality"
            ;;
        performance)
            cmd="$cmd tests/ -m performance"
            ;;
        quick)
            cmd="$cmd tests/ -x --tb=short --disable-warnings"
            COVERAGE=false
            ;;
        all|*)
            cmd="$cmd tests/"
            ;;
    esac

    # 添加覆盖率选项
    if [ "$COVERAGE" = true ]; then
        cmd="$cmd --cov=src --cov-report=term-missing"

        if [ "$HTML_REPORT" = true ]; then
            cmd="$cmd --cov-report=html:htmlcov"
        fi
    fi

    # 添加并行选项
    if [ "$PARALLEL" = true ]; then
        cmd="$cmd -n auto"
    fi

    # 添加详细输出
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    fi

    # 添加HTML报告
    if [ "$HTML_REPORT" = true ]; then
        cmd="$cmd --html=test-report.html --self-contained-html"
    fi

    # 添加JUnit XML报告
    cmd="$cmd --junit-xml=test-results.xml"

    echo $cmd
}

# 运行本地测试
run_local_tests() {
    echo -e "${BLUE}设置Python ${PYTHON_VERSION}环境...${NC}"

    # 检查Python是否安装
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 未安装${NC}"
        exit 1
    fi

    # 安装依赖
    echo -e "${BLUE}安装测试依赖...${NC}"
    pip install -q pytest pytest-cov pytest-xdist pytest-asyncio

    if [ -f requirements-test.txt ]; then
        pip install -q -r requirements-test.txt
    fi

    if [ -f requirements.txt ]; then
        pip install -q -r requirements.txt
    fi

    # 构建并运行pytest命令
    local pytest_cmd=$(build_pytest_command)
    echo -e "${GREEN}运行测试: $pytest_cmd${NC}"

    # 创建测试报告目录
    mkdir -p test-reports

    # 运行测试
    eval $pytest_cmd
}

# 运行Docker测试
run_docker_tests() {
    echo -e "${BLUE}使用Docker运行测试...${NC}"

    # 检查Docker是否安装
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker 未安装${NC}"
        exit 1
    fi

    # 选择Docker Compose服务
    local service_name="test-runner"
    case $TEST_TYPE in
        unit)
            service_name="unit-tests"
            ;;
        integration)
            service_name="integration-tests"
            ;;
        data-quality)
            service_name="data-quality-tests"
            ;;
        performance)
            service_name="performance-tests"
            ;;
    esac

    # 构建并运行Docker测试
    echo -e "${BLUE}构建Docker镜像...${NC}"
    docker-compose -f docker-compose.test.yml build $service_name

    echo -e "${BLUE}运行Docker测试...${NC}"
    docker-compose -f docker-compose.test.yml up --abort-on-container-exit $service_name

    # 获取测试结果
    local exit_code=$(docker-compose -f docker-compose.test.yml ps -q $service_name | xargs docker inspect -f '{{.State.ExitCode}}')

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ 所有测试通过!${NC}"
    else
        echo -e "${RED}❌ 测试失败，退出码: $exit_code${NC}"
        exit $exit_code
    fi

    # 清理
    docker-compose -f docker-compose.test.yml down
}

# 主执行逻辑
echo -e "${BLUE}=== levAnalyze 测试运行器 ===${NC}"
echo -e "${BLUE}测试类型: ${YELLOW}$TEST_TYPE${NC}"
echo -e "${BLUE}Python版本: ${YELLOW}$PYTHON_VERSION${NC}"
echo -e "${BLUE}覆盖率: ${YELLOW}$COVERAGE${NC}"
echo -e "${BLUE}并行执行: ${YELLOW}$PARALLEL${NC}"
echo -e "${BLUE}Docker模式: ${YELLOW}$DOCKER${NC}"
echo ""

# 记录开始时间
start_time=$(date +%s)

# 运行测试
if [ "$DOCKER" = true ]; then
    run_docker_tests
else
    run_local_tests
fi

# 计算耗时
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo -e "${GREEN}=== 测试完成 ===${NC}"
echo -e "${GREEN}总耗时: ${YELLOW}${duration}秒${NC}"

# 如果有HTML报告，显示访问信息
if [ "$HTML_REPORT" = true ]; then
    echo -e "${BLUE}HTML报告: ${YELLOW}test-report.html${NC}"
    if [ "$COVERAGE" = true ]; then
        echo -e "${BLUE}覆盖率报告: ${YELLOW}htmlcov/index.html${NC}"
    fi
fi

echo -e "${GREEN}✅ 所有任务完成!${NC}"