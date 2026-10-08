"""PoCL 那一层的测试：nnrun 只调标准的 OpenCL 接口与内建内核 pocl.dnn.dense.relu.i8，经 PoCL 的 tnpu 设备、TCP 进这里，
由测试台在引擎的总线口上逐个读写寄存器。被测的是引擎本体与适配模块（tqvp_sohaib_npu.v、npu_mmio.v），与片上接的是同一份；
整片的接法与引擎对 soft_fc 的逐项比另在整片测试的 npu 那一项里。nnrun 自己与 soft_fc 比，退出码不是 0 就不过。
"""
import os
import select
import socket
import subprocess

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

BASE = 0x1070_0000
NNRUN = os.environ["NNRUN"]


async def access(dut, op: bytes, off: int, v: int) -> int:
    dut.bus_addr_i.value = BASE + off
    dut.bus_wstrb_i.value = 0xF if op == b"w" else 0
    dut.bus_wdata_i.value = v
    dut.bus_valid_i.value = 1
    for n in range(10_000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if dut.bus_ready_o.value:
            break
    else:
        raise AssertionError(f"寄存器 0x{off:02x} 没有应答")
    got = int(dut.bus_rdata_o.value)
    await RisingEdge(dut.clk)
    dut.bus_valid_i.value = 0
    await RisingEdge(dut.clk)
    return got


async def nnrun(dut, *args: str) -> str:
    srv = socket.create_server(("127.0.0.1", 0))
    srv.setblocking(False)
    env = os.environ | {"POCL_DEVICES": "tnpu", "POCL_TNPU0_PARAMETERS": f"tcp:127.0.0.1:{srv.getsockname()[1]}"}
    p = subprocess.Popen([NNRUN, *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    conn, buf, n = None, b"", 0
    try:
        while True:
            if conn is None and select.select([srv], [], [], 0)[0]:
                conn, _ = srv.accept()
            if conn is not None and select.select([conn], [], [], 0)[0]:
                got = conn.recv(1 << 16)
                buf += got
                while len(buf) >= 10:
                    f, buf = buf[4:10], buf[10:]
                    v = await access(dut, f[0:1], f[1], int.from_bytes(f[2:6], "big"))
                    conn.sendall(bytes([0, 0, 0, 6]) + f[0:2] + v.to_bytes(4, "big"))
                    n += 1
                if not got:
                    conn.close()
                    conn = None
            if conn is None and p.poll() is not None:
                break
            await RisingEdge(dut.clk)
    finally:
        srv.close()
    out = p.stdout.read()
    dut._log.info("nnrun %s：%d 次寄存器访问\n%s", " ".join(args), n, out)
    assert p.returncode == 0, f"nnrun {' '.join(args)} 退出码 {p.returncode}"
    return out


async def up(dut):
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    dut.resetn.value = 0
    dut.bus_valid_i.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.resetn.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def info(dut):
    """设备认出引擎（标识寄存器），报出 PoCL 的内建内核名"""
    await up(dut)
    out = await nnrun(dut, "info")
    assert "to2610-npu" in out and "pocl.dnn.dense.relu.i8" in out


@cocotb.test()
async def mnist(dut):
    """随片的 MNIST 模型两层都走这个内建内核，与 soft_fc 逐项相同，八张都认对"""
    await up(dut)
    out = await nnrun(dut, "mnist")
    assert "不同 0 格，分类错 0 张" in out


@cocotb.test()
async def random_layers(dut):
    """随机的二十层（输入 1 至 144、输出 1 至 18），整层一次起跑，与 soft_fc 逐项相同"""
    await up(dut)
    out = await nnrun(dut, "random", "20")
    assert "不同 0 层" in out
