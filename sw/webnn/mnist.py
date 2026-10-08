"""随片的 MNIST 模型用 WebNN 建图、编译、计算（W3C 的 MLGraphBuilder，实现是 rustnn 的 pywebnn）。

每一层照引擎的算法写：int4 的输入与权相乘累加加偏置（这些数在 float32 里是精确的），ReLU，
再在 int32 上乘以乘数、右移（向下取整，负数也是）、加零点、夹到 −8..7。不涉及引擎寄存器，同一张图换个 WebNN 实现照样算。

    python mnist.py sw/npu/model.h                 八张图的十个输出与 model.h 里的 digit_want 逐项比，打出分类
    python mnist.py sw/npu/model.h --fc <fc 程序>  再随机出两百层，与 npu.h 的 soft_fc 逐项比；改一个权要不同
"""
import argparse
import pathlib
import random
import re
import subprocess
import sys

import numpy as np
import webnn


def arrays(header: str) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    def arr(name):
        m = re.search(rf"\b{name}\[\d+\] = \{{(.*?)\}};", header, re.S)
        if not m:
            sys.exit(f"model.h 里没有 {name}")
        return np.array([int(v) for v in m.group(1).split(",")])

    layers = []
    for spec in re.findall(r"\{(\d+), (\d+), (\d), (-?\d+), mlp_w(\d)", header):
        k, n, relu, zp, i = map(int, spec)
        layers.append(dict(k=k, n=n, relu=relu, zp=zp, w=arr(f"mlp_w{i}").reshape(k, n), b=arr(f"mlp_b{i}"),
                           q=arr(f"mlp_q{i}"), s=arr(f"mlp_s{i}")))
    return layers, arr("digit_px").reshape(-1, layers[0]["k"]), arr("digit_want").reshape(-1, layers[-1]["n"]), arr("digit_label")


def fc(g, h, L):
    """一层全连接，h 是 [批, k] 的 float32，回 [批, n] 的 float32（值是 −8..7 的整数）。"""
    n = L["n"]
    c32 = lambda v: g.constant(np.broadcast_to(np.asarray(v, np.int32), (n,)).copy())  # noqa: E731
    acc = g.add(g.matmul(h, g.constant(L["w"].astype(np.float32))), g.constant(L["b"].astype(np.float32)))
    if L["relu"]:
        acc = g.relu(acc)
    q = g.mul(g.cast(acc, "int32"), c32(L["q"]))
    d = c32(1 << np.asarray(L["s"]))
    # div 是向零截断，负数先减 d − 1 才是向下取整，与 C 里有符号数的算术右移相同
    floor = g.where_(g.lesser(q, c32(0)), g.div(g.add(q, c32(1 - (1 << np.asarray(L["s"])))), d), g.div(q, d))
    return g.cast(g.clamp(g.add(floor, c32(L["zp"])), min_value=-8, max_value=7), "float32")


def run(layers, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ctx = webnn.ML().create_context(device_type="cpu")
    g = ctx.create_graph_builder()
    h = g.input("x", list(x.shape), "float32")
    for L in layers:
        h = fc(g, h, L)
    graph = g.build({"out": g.cast(h, "int32"), "cls": g.arg_max(h, 1)})
    r = ctx.compute(graph, {"x": x.astype(np.float32)})
    return np.asarray(r["out"]).reshape(x.shape[0], -1), np.asarray(r["cls"]).reshape(-1).astype(int)


def soft(prog: str, L, x) -> list[int]:
    nums = [L["k"], L["n"], L["relu"], L["zp"], *L["w"].reshape(-1), *L["b"], *L["q"], *L["s"], *x]
    r = subprocess.run([prog], input=" ".join(map(str, nums)), capture_output=True, text=True, check=True)
    return [int(v) for v in r.stdout.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--fc", help="sw/webnn/fc.c 编出来的程序")
    a = ap.parse_args()
    layers, px, want, label = arrays(pathlib.Path(a.model).read_text(encoding="utf-8"))
    got, cls = run(layers, px)
    bad = int((got != want).sum())
    print(f"MNIST：{len(px)} 张图、每张 {want.shape[1]} 个输出，与 digit_want 不同 {bad} 格；分类 {cls.tolist()}，标签 {label.tolist()}")
    ok = bad == 0 and (cls == label).all()
    if a.fc:
        rng = random.Random(2610)
        diff = 0
        for _ in range(200):
            k, n = rng.randrange(1, 145), rng.randrange(1, 19)
            L = dict(k=k, n=n, relu=rng.randrange(2), zp=rng.randrange(-8, 8),
                     w=np.array([rng.randrange(-8, 8) for _ in range(k * n)]).reshape(k, n),
                     b=np.array([rng.randrange(-1000, 1001) for _ in range(n)]),
                     q=np.array([rng.randrange(1, 65536) for _ in range(n)]),
                     s=np.array([rng.randrange(8, 24) for _ in range(n)]))
            x = np.array([rng.randrange(-8, 8) for _ in range(k)])
            diff += run([L], x[None, :])[0][0].tolist() != soft(a.fc, L, x)
        L = dict(layers[0])
        L["w"] = L["w"].copy()
        L["w"][np.unravel_index(int(np.abs(L["w"]).argmax()), L["w"].shape)] *= -1
        moved = run([L, layers[1]], px)[0].tolist() != soft_chain(a.fc, layers, px)
        print(f"随机两百层：与 soft_fc 不同 {diff} 层；改一个权之后与原模型不同：{moved}")
        ok = ok and diff == 0 and moved
    sys.exit(0 if ok else 1)


def soft_chain(prog, layers, px) -> list[list[int]]:
    out = []
    for x in px:
        for L in layers:
            x = soft(prog, L, x)
        out.append(x)
    return out


if __name__ == "__main__":
    main()
