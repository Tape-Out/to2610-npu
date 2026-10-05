/* 引擎的裸机测试，由引导程序搬进 SDRAM 后执行。四步：认出引擎；累加器写了读回；
 * 随机的一层全连接与软件基线逐项比；两层 MLP 认几张 MNIST，与 numpy 基线、软件基线都逐项比。
 */
#include <stdint.h>

#include "npu.h"
#include "model.h"

#define REG(a) (*(volatile uint32_t *)(a))
#define LSR (*(volatile uint8_t *)0x10000005)

static void putch(char c) {
  while (!(LSR & 0x60)) {}
  REG(0x10000000) = c;
}

static void say(const char *s) {
  while (*s) putch(*s++);
}

static void hex(uint32_t v) {
  for (int i = 28; i >= 0; i -= 4) putch("0123456789abcdef"[v >> i & 15]);
}

static void dec(uint32_t v) {
  if (v >= 10) dec(v / 10);
  putch('0' + v % 10);
}

static uint32_t seed = 0x2610;
static uint32_t rnd(void) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}

void main(void) {
  volatile uint32_t *r = (volatile uint32_t *)NPU_BASE;

  say("npu id ");
  hex(r[NPU_ID]);
  say("\n");

  r[NPU_CTRL] = 0;
  r[NPU_ACC] = 0x8001;
  uint32_t a = r[NPU_ACC];
  r[NPU_ACC] = 0x1234;
  uint32_t b = r[NPU_ACC];
  say(a == 0xffff8001 && b == 0x1234 ? "acc ok\n" : "acc BAD\n");

  int good = 0;
  for (int t = 0; t < 20; t++) {
    int8_t in[4], w[4], hw, sw;
    for (int i = 0; i < 4; i++) in[i] = (int8_t)(rnd() % 16) - 8, w[i] = (int8_t)(rnd() % 16) - 8;
    int16_t bias = (int16_t)rnd();
    uint16_t qmul = rnd() % 32767;
    uint8_t shamt = rnd() % 31;
    struct fc l = {4, 1, rnd() & 1, (int)(rnd() % 16) - 8, w, &bias, &qmul, &shamt};
    npu_fc(r, &l, in, &hw);
    soft_fc(&l, in, &sw);
    good += hw == sw;
  }
  say("fc ok ");
  dec(good);
  say("/20\n");

  good = 0;
  for (int d = 0; d < DIGITS; d++) {
    int8_t h1[18], s1[18], hw[10], sw[10];
    npu_fc(r, &mlp[0], &digit_px[d * 144], h1);
    npu_fc(r, &mlp[1], h1, hw);
    soft_fc(&mlp[0], &digit_px[d * 144], s1);
    soft_fc(&mlp[1], s1, sw);
    int same = 1, top = 0;
    for (int i = 0; i < 10; i++) {
      same &= hw[i] == sw[i] && hw[i] == digit_want[d * 10 + i];
      if (hw[i] > hw[top]) top = i;
    }
    say("digit ");
    dec(digit_label[d]);
    say(" -> ");
    dec(top);
    say(same && top == digit_label[d] ? "\n" : " BAD\n");
    good += same && top == digit_label[d];
  }
  say("mnist ok ");
  dec(good);
  putch('/');
  dec(DIGITS);
  say("\ndone\n");
  for (;;) {}
}
