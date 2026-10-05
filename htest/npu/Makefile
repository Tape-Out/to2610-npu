O ?= build
CROSS ?= riscv64-unknown-elf-

$(O)/npu.bin: start.S main.c link.ld ../../sw/npu/npu.h ../../sw/npu/model.h
	mkdir -p $(O)
	$(CROSS)gcc -march=rv32ima_zicsr -mabi=ilp32 -O2 -ffreestanding -nostdlib -I../../sw/npu -Wl,-T,link.ld \
	  -o $(O)/npu.elf start.S main.c -lgcc
	$(CROSS)objcopy -O binary $(O)/npu.elf $@
