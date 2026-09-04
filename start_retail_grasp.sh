#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 项目 ROS setup 内部可能读取未定义变量，因此不能使用 set -u。
source "$PROJECT_DIR/setup_ros2_debug.sh" >/dev/null
export PYTHONNOUSERSITE=0

if ! python3 -c "import yaml, fastapi, uvicorn" >/dev/null 2>&1; then
  echo "缺少网页后端依赖：PyYAML、FastAPI 或 Uvicorn" >&2
  echo "请执行：python3 -m pip install --user PyYAML fastapi uvicorn" >&2
  exit 1
fi

exec python3 "$PROJECT_DIR/retail_grasp/retail_grasp_web.py" "$@"
