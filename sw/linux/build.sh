#!/usr/bin/env bash
# 编 to2610-npu 的镜像：to2610-kvc 的那一份，加引擎的驱动、设备树结点与 mnist 程序。
# 编法照 to2610-kvc 的 sw/linux/build.sh，这里只是把多出来的几样经环境变量交给它。
# 用法：build.sh <to2610-kvc 仓> [<kianriscv 仓>]；宿主的 GCC 到了 15 就把 DOCKER=1 带上
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$H/../.." && pwd)
L=$(realpath "${1:?要 to2610-kvc 仓的位置}")
K=$(realpath "${2:-$L/../kianriscv}")
export TO2610_DTS=$H/to2610-npu.dts
export TO2610_KCONFIG=$H/linux.fragment
export TO2610_KHOOK=$H/khook.sh
export TO2610_POST=$H/post.sh
export TO2610_OUT=$R/build/linux
export TO2610_MOUNT="$R ${TO2610_MOUNT:-}"
# to2610-kvc 编过的话接着用它那棵 buildroot：工具链与下载的源码包都在里面，只重编内核、OpenSBI 与根文件系统
[ ! -f "$L/build/linux/buildroot/Makefile" ] || export TO2610_BUILDROOT=$L/build/linux/buildroot
s=build.sh
[ -z "${DOCKER:-}" ] || s=docker.sh
bash "$L/sw/linux/$s" "$K"
