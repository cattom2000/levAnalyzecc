# levAnalyze 测试 Makefile
# 提供便捷的测试命令

.PHONY: help test test-unit test-integration test-data-quality test-performance test-quick
.PHONY: test-docker test-all clean lint format check install-dev
.PHONY: coverage-report docker-build docker-test docker-up docker-down

# 默认目标
help:
	@echo "levAnalyze 测试命令:"
	@echo ""
	@echo "测试命令:"
	@echo "  test              运行所有测试"
	@echo "  test-unit         运行单元测试"
	@echo "  test-integration  运行集成测试"
	@echo "  test-data-quality 运行数据质量测试"
	@echo "  test-performance  运行性能测试"
	@echo "  test-quick        运行快速检查"
	@echo "  test-docker       使用Docker运行所有测试"
	@echo "  test-all          运行完整测试套件"
	@echo ""
	@echo "代码质量:"
	@echo "  lint              代码风格检查"
	@echo "  format            代码格式化"
	@echo "  check             快速语法检查"
	@echo ""
	@echo "环境管理:"
	@echo "  install-dev       安装开发依赖"
	@echo "  clean             清理临时文件"
	@echo "  coverage-report   生成覆盖率报告"
	@echo ""
	@echo "Docker命令:"
	@echo "  docker-build      构建测试镜像"
	@echo "  docker-up         启动测试环境"
	@echo "  docker-down       停止测试环境"

# 环境变量
PYTHON := python3
PIP := pip3
PYTEST := pytest

# 测试命令
test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --junit-xml=test-results.xml

test-unit:
	$(PYTHON) -m pytest tests/ -m unit -v --junit-xml=unit-test-results.xml

test-integration:
	$(PYTHON) -m pytest tests/ -m integration -v --junit-xml=integration-test-results.xml

test-data-quality:
	$(PYTHON) -m pytest tests/ -m data_quality -v --junit-xml=data-quality-test-results.xml

test-performance:
	$(PYTHON) -m pytest tests/ -m performance -v --benchmark-json=performance-results.json

test-quick:
	$(PYTHON) -m pytest tests/ -x --tb=short --disable-warnings -q

test-docker:
	docker-compose -f docker-compose.test.yml up --abort-on-container-exit test-runner

test-all: clean lint test-unit test-integration test-data-quality test-performance coverage-report
	@echo "✅ 完整测试套件执行完成"

# 代码质量命令
lint:
	@echo "运行代码质量检查..."
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/
	mypy src/ --ignore-missing-imports

format:
	@echo "格式化代码..."
	black src/ tests/
	isort src/ tests/

check:
	@echo "快速语法检查..."
	$(PYTHON) -m py_compile src/**/*.py tests/**/*.py

# 环境管理
install-dev:
	@echo "安装开发依赖..."
	$(PIP) install --upgrade pip
	if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; fi
	if [ -f requirements-test.txt ]; then $(PIP) install -r requirements-test.txt; fi
	$(PIP) install -e .

clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ test-results.xml .pytest_cache/
	rm -rf test-reports/ *.html

coverage-report: test
	@echo "生成HTML覆盖率报告..."
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html:htmlcov --cov-report=xml
	@echo "覆盖率报告: htmlcov/index.html"

# Docker命令
docker-build:
	docker-compose -f docker-compose.test.yml build

docker-up:
	docker-compose -f docker-compose.test.yml up -d

docker-down:
	docker-compose -f docker-compose.test.yml down

# 开发工作流
dev-setup: install-dev
	@echo "开发环境设置完成"

dev-test: test-quick lint
	@echo "开发测试完成"

ci-test: clean lint test-unit
	@echo "CI测试完成"

# 性能分析
profile:
	@echo "运行性能分析..."
	$(PYTHON) -m cProfile -o profile.stats -m pytest tests/ -k performance
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# 内存分析
memory-profile:
	@echo "运行内存分析..."
	mprof run $(PYTHON) -m pytest tests/ -m performance
	mprof plot

# 安全检查
security:
	@echo "运行安全检查..."
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

# 完整CI管道
ci-pipeline: clean security lint test-unit test-integration test-data-quality
	@echo "CI管道执行完成"

# 依赖检查
deps-check:
	@echo "检查依赖安全性..."
	$(PIP) audit
	safety check

# 文档测试
docs-test:
	@echo "测试文档..."
	if [ -d docs ]; then $(MAKE) -C docs html; fi
	$(PYTHON) -m pytest --doctest-modules src/

# 数据验证
validate-test-data:
	@echo "验证测试数据..."
	$(PYTHON) -c "
from tests.fixtures.data.generators import MockDataGenerator
import pandas as pd

print('测试Mock数据生成器...')
finra_data = MockDataGenerator.generate_finra_margin_data(periods=12)
sp500_data = MockDataGenerator.generate_sp500_data(periods=30)
fred_data = MockDataGenerator.generate_fred_data(periods=6)

print('✅ 所有测试数据生成正常')
"

# 性能基准
benchmark:
	@echo "运行性能基准..."
	$(PYTHON) -m pytest tests/ -m performance --benchmark-only --benchmark-sort=mean

# 多Python版本测试
test-multi-version:
	@echo "测试多个Python版本..."
	for version in 3.9 3.10 3.11 3.12; do \
		if command -v python$$version &> /dev/null; then \
			echo "测试Python $$version..."; \
			python$$version -m pytest tests/ -x --tb=short || true; \
		else \
			echo "Python $$version 未安装，跳过"; \
		fi; \
	done

# 并行测试
test-parallel:
	$(PYTHON) -m pytest tests/ -n auto --dist=loadfile

# 监视模式(测试驱动开发)
watch:
	$(PYTHON) -m pytest tests/ -f

# 统计信息
stats:
	@echo "项目统计:"
	@echo "源文件数量: $$(find src/ -name "*.py" | wc -l)"
	@echo "测试文件数量: $$(find tests/ -name "*.py" | wc -l)"
	@echo "代码行数: $$(find src/ -name "*.py" -exec wc -l {} + | tail -1)"
	@echo "测试行数: $$(find tests/ -name "*.py" -exec wc -l {} + | tail -1)"

# 帮助信息(默认)
.DEFAULT_GOAL := help