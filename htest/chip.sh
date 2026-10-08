#!/usr/bin/env bash
# 整片测试，全部跑在 ran asic 出的那份 .v 上。测试台、片外模型来自黑盒仓，引导程序与拼镜像的脚本来自 to2610-kvc。
#   hello  to2610-kvc 的裸机冒烟原样跑一遍：加了引擎之后原有的东西不许变
#   periph to2610-kvc 的外设测试原样跑一遍：GPIO、两路 SPI、计时器中断、PLIC、重启
#   isa    标准 riscv-tests 逐个跑，同 to2610-kvc：加了引擎之后核的行为不许变
#   arch-test  riscv-arch-test 的非特权部分，同 to2610-kvc
#   boot   引导程序带回显载荷，同上
#   npu    引擎的裸机测试：认出引擎、累加器读写、随机全连接与软件逐项比、两层 MLP 认八张 MNIST
#   image  载荷里的内核带着根文件系统先在 QEMU 上起到 shell（to2610-kvc 的那一关）：镜像坏了一分钟就知道
#   linux  起到 shell 后跑 mnist：经 /dev/npu 用引擎认同样八张，与软件基线逐项比
# 用法：chip.sh <输出目录> <gf180mcu-kianv-rv32ima-sv32 仓> <to2610-kvc 仓>。已经跑过 ran asic 的，把输出目录给 CHIP_ASIC。
# Linux 镜像先用 sw/linux/build.sh 编，位置给 LINUX_IMAGE（默认 build/linux/fw_payload.bin）；没编过就取发布页上的那一份
set -euo pipefail
cd "$(dirname "$0")/.."
O=$(realpath -m "$1")
K=$(realpath "$2")
L=$(realpath "$3")
rm -rf "$O"
mkdir -p "$O"
mkdir -p build
A=${CHIP_ASIC:-$O/asic}
[ -s "$A/report.json" ] || $XIRANG asic to2610-npu --no-run -o "$A"
top=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['top'])" "$A/report.json")
python3 "$L/htest/shim.py" "$A/report.json" > "$O/tb.v"
bash "$K/htest/sim.sh" "$O/sim" "$O/tb.v" "$A/$top.v"

res=()
run() {
  local name=$1 what=$2 t0=$SECONDS rc=0
  shift 2
  "$O/sim/Vtb" "$@" > "$O/$name.log" 2> "$O/$name.err" || rc=$?
  tail -n 3 "$O/$name.err"
  res+=("$name=$rc:$((SECONDS - t0)):$what")
}

make -s -C "$K/htest/hello" O="$O/hello"
run hello "to2610-kvc 的裸机冒烟：从 Flash 就地执行，读写 SDRAM" \
  +flash="$O/hello/hello.bin@0x100000" +script="$K/htest/hello/script" +max=30000000

make -s -C "$K/htest/periph" O="$O/periph"
run periph "to2610-kvc 的外设测试：GPIO、两路 SPI 与回声从设备、计时器中断、PLIC、重启" \
  +flash="$O/periph/periph.bin@0x100000" +script="$K/htest/periph/script" +spiecho +max=30000000

t0=$SECONDS
rc=0
SIM="$O/sim" bash "$K/htest/isa.sh" "$O/isa" > "$O/isa.log" 2>&1 || rc=$?
tail -n 3 "$O/isa.log"
res+=("isa=$rc:$((SECONDS - t0)):riscv-tests 的 rv32ui、um、ua、mi、si 逐个在这份 .v 上跑，程序放进 SDRAM、测试台盯 tohost；79 个全过，不适用的 5 个（Zacas、硬件非对齐访存、PMP、调试触发器）不跑")

t0=$SECONDS
rc=0
SIM="$O/sim" bash "$K/htest/arch-test.sh" "$O/arch-test" > "$O/arch-test.log" 2>&1 || rc=$?
tail -n 3 "$O/arch-test.log"
res+=("arch-test=$rc:$((SECONDS - t0)):riscv-arch-test（ACT4）的 I、M、Zmmul、Zaamo、Zalrsc、Zicsr、Zifencei、Zicntr 共 71 个自检程序逐个在这份 .v 上跑，期望值出自 Sail 模型")

make -s -C "$L/sw/boot" O="$O/boot"
make -s -C "$L/htest/echo" O="$O/echo"
python3 "$L/sw/pack.py" "$O/boot/boot.bin" "$O/echo/echo.bin" "$O/echo.flash"
run boot "引导程序搬载荷并核对校验和，载荷回显串口" \
  +flash="$O/echo.flash@0" +script="$L/htest/echo/script" +max=60000000

make -s -C htest/npu O="$O/npu"
python3 "$L/sw/pack.py" "$O/boot/boot.bin" "$O/npu/npu.bin" "$O/npu.flash"
run npu "引擎的裸机测试：标识寄存器、累加器读写、20 组随机全连接、两层 MLP 认八张 MNIST，与软件及 numpy 基线逐项相同" \
  +flash="$O/npu.flash@0" +script=htest/npu/script +max=300000000

I=${LINUX_IMAGE:-build/linux/fw_payload.bin}
# 没在本机编过（流水线上）：取发布页上的那一份，地址与摘要钉在 sw/linux/image.pin，摘要对不上就不用
if [ ! -s "$I" ] && [ -s sw/linux/image.pin ]; then
  read -r url sum < sw/linux/image.pin
  mkdir -p "$(dirname "$I")"
  if curl -fsSL --retry 5 -o "$I.part" "$url" && echo "$sum  $I.part" | sha256sum -c - > /dev/null; then
    mv "$I.part" "$I"
  else
    rm -f "$I.part"
    echo "取不到镜像 $url，或摘要不是 $sum"
  fi
fi
if [ -s "$I" ]; then
  t0=$SECONDS
  rc=0
  python3 "$L/htest/qemu.py" "$I" "$L/htest/image.script" "$O/qemu" > "$O/image.log" 2>&1 || rc=$?
  tail -n 3 "$O/image.log"
  res+=("image=$rc:$((SECONDS - t0)):载荷里的内核带着根文件系统在 QEMU 的 virt 机器上起到 shell；只验软件，拦坏镜像")
  if [ $rc = 0 ]; then
    python3 "$L/sw/pack.py" "$O/boot/boot.bin" "$I" "$O/linux.flash"
    # 发字节的间隔 60 万拍（12 毫秒）：串口没有接收缓冲，内核往控制台打一行要关中断五毫秒多，
    # 间隔比它短，那一行里到的第二个字就丢了
    run linux "从 Flash 引导 Linux，驱动认出引擎，mnist 经 /dev/npu 认八张图，与软件基线逐项相同" \
      +flash="$O/linux.flash@0" +script=htest/linux.script +pace=600000 +max="${LINUX_MAX:-8000000000}" +beat=1000000000
  else
    res+=("linux=1:0:镜像没过 QEMU 那一关，没跑")
  fi
else
  echo "没有 Linux 镜像 $I：第四段没跑（先跑 sw/linux/build.sh）"
  res+=("image=1:0:没有镜像，没跑" "linux=1:0:没有镜像，没跑")
fi

python3 "$L/htest/junit.py" "$O/results.xml" "${res[@]}"
printf '%s\n' "${res[@]}"
if grep -q '<failure' "$O/results.xml"; then echo "有用例没过"; exit 1; fi
echo "整片测试 ${#res[@]} 段全过"
