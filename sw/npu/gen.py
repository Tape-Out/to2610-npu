"""出 sw/npu/model.h：上游 TinyQV 仓里那份两层 MLP 的权重，加几张 MNIST 测试图与 numpy 基线的结果。

模型、预处理（28×28 双线性缩到 12×12、量化到 int4）与基线都直接用上游测试的 utils.py，不另写一份。
用法：gen.py <ttsky25a-tinyqv 仓> <放 MNIST 的目录> [张数]
"""
import pathlib
import sys

import numpy as np

T = pathlib.Path(sys.argv[1]) / "third_party/ttsky25a-tinyqv/test"
sys.path.insert(0, str(T))
from user_peripherals.npu.utils import NumpyModel, dtype_to_bounds, fetch_mnist, resize_bilinear  # noqa: E402

N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
A = T / "user_peripherals/npu/assets"
model = NumpyModel(A / "model.json", A / "tensors.bin")
_, _, x_test, y_test = fetch_mnist(sys.argv[2])
x = resize_bilinear((x_test.astype(np.float32) / 255.0).reshape(-1, 28, 28), 12, 12)
qmin, qmax = dtype_to_bounds(4, True)
x = (x / (1.0 / (qmax - qmin)) + qmin).astype(np.int8)
y = model(x)
hit = y.argmax(axis=-1) == y_test
print(f"基线 {hit.sum()}/{len(y_test)} 对，{100.0 * hit.mean():.1f}%", file=sys.stderr)
# 每个数字取基线认对的头一张
pick = [int(np.flatnonzero(hit & (y_test == d))[0]) for d in range(10)][:N]


def arr(ctype, name, a):
    v = np.asarray(a).reshape(-1)
    rows = [", ".join(str(int(t)) for t in v[i:i + 24]) for i in range(0, len(v), 24)]
    return f"static const {ctype} {name}[{len(v)}] = {{\n  " + ",\n  ".join(rows) + "\n};\n"


out = ["/* 由 sw/npu/gen.py 生成，不要手改。两层全连接：144 入 18 出带 ReLU，18 入 10 出 */\n"]
fcs = [op for op in model.ops if op["op"] == "fully_connected"]
rows = []
for i, (op, layer) in enumerate(zip(fcs, [f for f in model.layers if hasattr(f, "keywords")]), 1):
    kw = layer.keywords
    w, b = kw["weight"], kw["bias"].astype(np.int16)
    out += [arr("int8_t", f"mlp_w{i}", w), arr("int16_t", f"mlp_b{i}", b),
            arr("uint16_t", f"mlp_q{i}", kw["qmul"]), arr("uint8_t", f"mlp_s{i}", kw["shamt"])]
    rows.append(f"  {{{w.shape[0]}, {w.shape[1]}, {int(op['act'] == 'RELU')}, {int(kw['output_zp'])}, "
                f"mlp_w{i}, mlp_b{i}, mlp_q{i}, mlp_s{i}}}")
out.append("static const struct fc mlp[] = {\n" + ",\n".join(rows) + "\n};\n")
out.append(f"#define DIGITS {len(pick)}\n")
# 图与基线结果都摊平：第 d 张的第 i 个像素在 digit_px[d * 144 + i]
out += [arr("int8_t", "digit_px", x[pick]), arr("uint8_t", "digit_label", y_test[pick]),
        arr("int8_t", "digit_want", y[pick])]
sys.stdout.write("".join(out))
