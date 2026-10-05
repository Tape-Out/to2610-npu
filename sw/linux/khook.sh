#!/usr/bin/env bash
# 把引擎的驱动放进内核源码树：源文件拷进 drivers/misc，Makefile 与 Kconfig 各追加一条。
# 追加在文件末尾，不依赖上下文，内核换版本也套得上。用法：khook.sh <内核源码目录>
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
L=$(realpath "$1")
cp "$H/to2610_npu.c" "$L/drivers/misc/"
grep -q TO2610_NPU "$L/drivers/misc/Makefile" || echo 'obj-$(CONFIG_TO2610_NPU) += to2610_npu.o' >> "$L/drivers/misc/Makefile"
grep -q TO2610_NPU "$L/drivers/misc/Kconfig" || cat >> "$L/drivers/misc/Kconfig" <<'K'

config TO2610_NPU
	tristate "MLP engine of the to2610-npu chip"
	depends on OF && HAS_IOMEM
	help
	  Maps the register page of the engine to user space as /dev/npu.
K
