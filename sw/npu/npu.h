/* to2610-npu 片上 MLP 引擎的寄存器与一层全连接的两种算法：经引擎算、纯软件算。
 * 裸机测试与 Linux 下的用户程序都用这一份，寄存器基址由调用方给（裸机是物理地址，Linux 下是 mmap 回来的）。
 */
#ifndef TO2610_NPU_H
#define TO2610_NPU_H
#include <stdint.h>

#define NPU_BASE 0x10700000u
#define NPU_SPAN 0x20u
#define NPU_MAGIC 0x4e505531u

enum { NPU_DAT, NPU_ACC, NPU_OUT, NPU_SHAMT, NPU_QMUL, NPU_CTRL, NPU_RSV, NPU_ID };

/* 一层全连接：k 个输入、n 个输出，都是 int4；权重按 [输入][输出] 排 */
struct fc {
  int k, n, relu, zp;
  const int8_t *w;
  const int16_t *bias;
  const uint16_t *qmul;
  const uint8_t *shamt;
};

/* 引擎一次乘加四对：输入不够四个的补零 */
static inline void npu_fc(volatile uint32_t *r, const struct fc *l, const int8_t *in, int8_t *out) {
  r[NPU_CTRL] = (uint32_t)(l->zp & 15) << 1 | (uint32_t)l->relu;
  for (int n = 0; n < l->n; n++) {
    r[NPU_SHAMT] = l->shamt[n];
    r[NPU_QMUL] = l->qmul[n];
    r[NPU_ACC] = (uint16_t)l->bias[n];
    for (int k = 0; k < l->k; k += 4) {
      uint32_t d = 0;
      for (int i = 0; i < 4 && k + i < l->k; i++)
        d |= (uint32_t)(in[k + i] & 15) << (4 * i) | (uint32_t)(l->w[(k + i) * l->n + n] & 15) << (16 + 4 * i);
      r[NPU_DAT] = d;
    }
    out[n] = (int8_t)r[NPU_OUT];
  }
}

/* 软件基线，与上游测试里的 qfc 逐步相同：16 位累加会回绕，缩放后向下取整，再夹到 int4 */
static inline void soft_fc(const struct fc *l, const int8_t *in, int8_t *out) {
  for (int n = 0; n < l->n; n++) {
    int16_t acc = l->bias[n];
    for (int k = 0; k < l->k; k++) acc = (int16_t)(acc + in[k] * l->w[k * l->n + n]);
    if (l->relu && acc < 0) acc = 0;
    int32_t v = (((int32_t)l->qmul[n] * acc) >> l->shamt[n]) + l->zp;
    out[n] = v < -8 ? -8 : v > 7 ? 7 : (int8_t)v;
  }
}

#endif
