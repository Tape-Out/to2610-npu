#!/usr/bin/env bash
# 拼出这一颗的源码树 build/src：从黑盒仓拼好的那份 KianV SoC 源码（上游加它的补丁）起，套上 patch/soc.patch（在片上总线上多接一个口），
# 再放进 TinyQV 仓里的引擎本体与本仓的适配模块。两个上游子模块都不动。
# 用法：setup.sh <gf180mcu-kianv-rv32ima-sv32 仓> <ttsky25a-tinyqv 仓>
set -euo pipefail
cd "$(dirname "$0")/.."
K=$(realpath "$1")
T=$(realpath "$2")/third_party/ttsky25a-tinyqv
rm -rf build/src
mkdir -p build
bash "$K/htest/setup.sh" > /dev/null
cp -r "$K/build/src" build/src
patch -s -p1 -d build < patch/soc.patch
cp "$T/src/user_peripherals/npu/peripheral.v" build/src/tqvp_sohaib_npu.v
cp hwsrc/npu_mmio.v build/src/
echo "build/src：$(find build/src -name '*.v' -o -name '*.sv' | wc -l) 个源文件"
