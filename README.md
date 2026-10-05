# to2610-npu

A Linux chip with an on-chip MLP engine, for the ECOS 2610 shuttle: the KianV RV32IMA Sv32 SoC of [`to2610-kvc`](https://github.com/Tape-Out/to2610-kvc) with the int4 multiply-accumulate engine of Tiny Tapeout's TinyQV on its bus. Linux drives the engine through `/dev/npu`.

![maturity](https://img.shields.io/badge/maturity-simulated-yellow) ![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0%20OR%20MulanPSL--2.0-blue)

Two upstream designs, their submodules untouched: the SoC from [`gf180mcu-kianv-rv32ima-sv32`](https://github.com/Tape-Out/gf180mcu-kianv-rv32ima-sv32) and the engine `tqvp_sohaib_npu` from [`ttsky25a-tinyqv`](https://github.com/Tape-Out/ttsky25a-tinyqv). What this repository adds is `hwsrc/npu_mmio.v`, which puts the engine on the KianV bus as a word-addressed register block, and `patch/soc.patch`, thirty lines that decode `0x1070_0000` in upstream's `soc.v`. The `src` task assembles the source tree under `build/src`; [`xirang`](https://github.com/Tape-Out/xirang) wraps it into the five-port top of MPC-Frame. The pins are those of `to2610-kvc`, bit for bit, so both chips use one board.

The functions, the registers, the board wiring, the chip tests and the limits are in [`docs/流片说明.md`](docs/流片说明.md); the tape-out report is generated from that file.

## The engine

| Offset | Register | Access | What |
|:--:|:--:|:--:|:--:|
| `0x00` | DAT | write | four int4 inputs in the low half, four int4 weights in the high half; one write, one multiply-accumulate |
| `0x04` | ACC | read, write | the 16-bit accumulator |
| `0x08` | OUT | read | the accumulator scaled to int4; takes six cycles |
| `0x0C` | SHAMT | write | shift, 5 bits |
| `0x10` | QMUL | write | quantised multiplier, 15 bits |
| `0x14` | CTRL | write | bit 0 ReLU, bits 4:1 output zero point |
| `0x1C` | ID | read | `0x4E505531` |

`sw/npu/npu.h` has a fully connected layer both ways, through the engine and in software; `sw/npu/model.h` is upstream's two-layer MLP for 12×12 MNIST with eight test images.

## Software

```console
$ DOCKER=1 bash sw/linux/build.sh ../to2610-kvc      # the to2610-kvc image plus the driver, the node and mnist
$ python3 ../to2610-kvc/sw/pack.py boot.bin build/linux/fw_payload.bin flash.bin
# mnist                                                 # on the chip: eight digits through /dev/npu, checked against software
```

`sw/linux/to2610_npu.c` is the kernel driver: it checks the ID register and maps the register page to user space as `/dev/npu`.

## Testing and tape-out

```console
$ ran run to2610-npu src                  # assemble build/src
$ ran test to2610-npu                     # the chip tests, on the Verilog file that goes to the shuttle
$ ran asic to2610-npu                     # to2610_npu.v, ecc at 50 MHz, report.json
```

The chip tests run on Verilator with pin-level models of the SDRAM, the flash and the UART: the two bare-metal tests of `to2610-kvc` unchanged, a bare-metal test of the engine against software and upstream's numpy baseline, and Linux running `mnist`.

## License

任选其一：

- [MIT](LICENSE-MIT)
- [Apache 2.0](LICENSE-APACHE)
- [木兰宽松许可证 第2版](LICENSE-MULAN)

`SPDX-License-Identifier: MIT OR Apache-2.0 OR MulanPSL-2.0`

`sw/linux/to2610_npu.c` goes into the Linux kernel and is GPL-2.0. The upstream sources keep their own licences: Apache-2.0 for both.

除非另行说明，你提交的贡献按上述三者同时授权，不附加其他条件。
