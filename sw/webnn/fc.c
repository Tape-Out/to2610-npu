// 主机上跑 sw/npu/npu.h 的 soft_fc（引擎在整片测试里与它逐项比过），给 WebNN 那一边当参照。
// 标准输入：k n relu zp，再 k×n 个权（按 [输入][输出]）、n 个偏置、n 个乘数、n 个移位、k 个输入；标准输出 n 个结果
#include <stdio.h>
#include <stdlib.h>

#include "../npu/npu.h"

static int get(void) {
  int v;
  if (scanf("%d", &v) != 1) {
    fputs("fc：输入不够\n", stderr);
    exit(2);
  }
  return v;
}

int main(void) {
  struct fc l = {0};
  l.k = get(), l.n = get(), l.relu = get(), l.zp = get();
  int8_t *w = malloc((size_t)(l.k * l.n)), *in = malloc((size_t)l.k), *out = malloc((size_t)l.n);
  int16_t *b = malloc(sizeof *b * (size_t)l.n);
  uint16_t *q = malloc(sizeof *q * (size_t)l.n);
  uint8_t *s = malloc((size_t)l.n);
  for (int i = 0; i < l.k * l.n; i++) w[i] = (int8_t)get();
  for (int i = 0; i < l.n; i++) b[i] = (int16_t)get();
  for (int i = 0; i < l.n; i++) q[i] = (uint16_t)get();
  for (int i = 0; i < l.n; i++) s[i] = (uint8_t)get();
  for (int i = 0; i < l.k; i++) in[i] = (int8_t)get();
  l.w = w, l.bias = b, l.qmul = q, l.shamt = s;
  soft_fc(&l, in, out);
  for (int i = 0; i < l.n; i++) printf("%d%c", out[i], i + 1 < l.n ? ' ' : '\n');
  return 0;
}
