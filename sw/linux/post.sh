#!/usr/bin/env bash
# buildroot 在打包根文件系统前调它：用刚编好的交叉编译器把 mnist 编进去。第一个参数是 TARGET_DIR
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
"$HOST_DIR/bin/riscv32-buildroot-linux-gnu-gcc" -O2 -static -Wall -I"$H/../npu" -o "$1/usr/bin/mnist" "$H/mnist.c"
