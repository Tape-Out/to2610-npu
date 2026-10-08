#!/usr/bin/env bash
# WebNN 那一层：随片的 MNIST 模型用 WebNN 建图计算，与 model.h 的基线、与 npu.h 的 soft_fc 逐项比（引擎与 soft_fc
# 在整片测试里比过）。WebNN 的实现取 rustnn 的 pywebnn，它只出 CPython 3.12 的轮子，由 uv 另起一个 3.12 的环境。
# 用法：webnn.sh <输出目录>
set -euo pipefail
cd "$(dirname "$0")/.."
O=$(realpath -m "$1")
rm -rf "$O"
mkdir -p "$O"
command -v uv > /dev/null || python3 -m pip install -q uv
uv venv -q -p 3.12 "$O/venv"
VIRTUAL_ENV=$O/venv uv pip install -q "pywebnn==0.5.12" numpy
cc -std=c23 -O2 -Wall -Wextra -Werror -o "$O/fc" sw/webnn/fc.c
"$O/venv/bin/python" sw/webnn/mnist.py sw/npu/model.h --fc "$O/fc" | tee "$O/webnn.log"
