#!/usr/bin/env bash
# PoCL 那一层的埋错：tnpu 设备里改掉一处，重编 PoCL、再跑 htest/pocl.sh，每一处都要红；原样要过。
# 用法：pocl-faults.sh <输出目录> <gf180mcu-kianv-rv32ima-sv32 仓> <ttsky25a-tinyqv 仓>
set -u
cd "$(dirname "$0")/.."
O=$(realpath -m "$1")
rm -rf "$O"
mkdir -p "$O"

# 名字~sed 的写法：权的位置、权的行、ReLU 的那一位、偏置取哪一个
MUTS='weightbits~s/<< (16 + 4 \* t)/<< (20 + 4 * t)/
weightrow~s/const int8_t \*wj = w + j \* k;/const int8_t *wj = w + j;/
relu~s/(uint32_t)(zp \& 15) << 1 | 1/(uint32_t)(zp \& 15) << 1/
bias~s/(uint16_t)bias\[j\]/(uint16_t)bias[0]/'

if ! TNPU_SRC=$PWD/sw/pocl/tnpu bash htest/pocl.sh "$O/clean" "$2" "$3" > "$O/clean.log" 2>&1; then
  tail -n 20 "$O/clean.log"
  echo "原样没过"
  exit 1
fi
bad=0
count=0
while IFS='~' read -r name expr; do
  count=$((count + 1))
  m=$O/src-$name
  cp -r sw/pocl/tnpu "$m"
  sed -i -e "$expr" "$m/tnpu.c"
  if cmp -s sw/pocl/tnpu/tnpu.c "$m/tnpu.c"; then
    echo "$name 没埋上"
    bad=1
    continue
  fi
  if TNPU_SRC=$m bash htest/pocl.sh "$O/$name" "$2" "$3" > "$O/$name.log" 2>&1; then
    echo "$name 埋了错还过了"
    bad=1
  else
    echo "$name 红了：$(grep -a '不同\|退出码' "$O/$name.log" | head -n 2 | tr '\n' ' ' | cut -c1-160)"
  fi
done <<< "$MUTS"
[ $bad = 0 ] || { echo "有埋错没被抓到"; exit 1; }
echo "PoCL 那一层：原样过，$count 处埋错都红"
