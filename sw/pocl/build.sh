#!/usr/bin/env bash
# 编一份带 tnpu 设备的 PoCL 与 nnrun。PoCL 取 pocl.pin 钉住的那一版（与 to2610-gpu 的 tgpu 同一版），
# 打上登记设备的补丁、把 tnpu/ 拷进去；不带 LLVM，设备只认内建内核 pocl.dnn.dense.relu.i8。
# 用法：build.sh <工作目录> <安装前缀>；nnrun 装在 <安装前缀>/bin
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
W=$(realpath -m "$1")
P=$(realpath -m "$2")
read -r tag sha < "$here/pocl.pin"
mkdir -p "$W"
if [ ! -d "$W/src/.git" ]; then
  rm -rf "$W/src"
  git clone -q --depth 1 --branch "$tag" https://github.com/pocl/pocl.git "$W/src"
fi
cd "$W/src"
got=$(git rev-parse HEAD)
[ "$got" = "$sha" ] || { echo "PoCL $tag 是 $got，钉的是 $sha"; exit 1; }
git checkout -q -- .
rm -rf lib/CL/devices/tnpu
git apply "$here/register.patch"
# PoCL 的 POCL_ARG_TYPE_MUTABLE 与 POCL_ARG_TYPE_NONE 都是 0：按名字建的内建内核只要有标量参数，复制元数据时就当成
# DBK 的可变参数去读一个空指针。只有 DBK 才给那张表，没给的照原样复制
git apply "$here/pod-args.patch"
# TNPU_SRC 给了就拷那一份（埋错时用）
cp -r "${TNPU_SRC:-$here/tnpu}" lib/CL/devices/tnpu
cmake -S . -B "$W/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" \
  -DENABLE_LLVM=OFF -DENABLE_HOST_CPU_DEVICES=OFF -DENABLE_TNPU_DEVICE=ON -DENABLE_ICD=OFF \
  -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_POCLCC=OFF -DENABLE_HWLOC=OFF > "$W/cmake.log" 2>&1
cmake --build "$W/build" > "$W/build.log"
cmake --install "$W/build" > /dev/null
mkdir -p "$P/bin"
cc -O2 -Wall -Werror -I "$W/src/include" -I "$here/../npu" "$here/nnrun.c" -L "$P/lib" -Wl,-rpath,"$P/lib" -lOpenCL -o "$P/bin/nnrun"
echo "PoCL $tag 带 tnpu 装在 $P"
