#!/usr/bin/env bash
# PoCL 那一层：编带 tnpu 设备的 PoCL 与 nnrun，拼源码树，在引擎本体加适配模块上用 cocotb 跑 htest/pocl/test_pocl.py。
# 用法：pocl.sh <输出目录> <gf180mcu-kianv-rv32ima-sv32 仓> <ttsky25a-tinyqv 仓>
set -euo pipefail
cd "$(dirname "$0")/.."
O=$(realpath -m "$1")
rm -rf "$O"
mkdir -p "$O"
bash sw/pocl/build.sh "$O/pocl" "$O/pocl/inst"
bash htest/setup.sh "$2" "$3"
NNRUN=$O/pocl/inst/bin/nnrun make -s -C htest/pocl -f "$(cocotb-config --makefiles)/Makefile.sim" SIM=icarus TOPLEVEL_LANG=verilog \
  VERILOG_SOURCES="$PWD/build/src/tqvp_sohaib_npu.v $PWD/build/src/npu_mmio.v" TOPLEVEL=npu_mmio MODULE=test_pocl \
  SIM_BUILD="$O/sim" COCOTB_RESULTS_FILE="$O/results.xml" > "$O/sim.log" 2>&1 || true
tail -n 30 "$O/sim.log"
# cocotb 失败时 make 照样返回 0，判据是结果文件
[ -s "$O/results.xml" ] || { echo "没有 results.xml"; exit 1; }
if grep -q '<failure' "$O/results.xml"; then echo "有用例没过"; exit 1; fi
n=$(grep -c '<testcase' "$O/results.xml")
[ "$n" -gt 0 ] || { echo "一个用例也没跑"; exit 1; }
echo "PoCL 那一层 $n 个全过"
