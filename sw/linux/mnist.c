/* 在 Linux 下经 /dev/npu 用片上的 MLP 引擎认几张 MNIST，同一层再用软件算一遍，逐项比并各计时。
 * 模型与测试图在 sw/npu/model.h 里，是上游 TinyQV 仓那份 12×12、4 位量化的两层全连接。
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "npu.h"
#include "model.h"

static long usec(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000L + t.tv_nsec / 1000;
}

int main(void) {
  int fd = open("/dev/npu", O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("/dev/npu");
    return 1;
  }
  volatile uint32_t *r = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (r == MAP_FAILED) {
    perror("mmap");
    return 1;
  }
  if (r[NPU_ID] != NPU_MAGIC) {
    fprintf(stderr, "no engine: id %08x\n", (unsigned)r[NPU_ID]);
    return 1;
  }

  int good = 0;
  long hw_us = 0, sw_us = 0;
  for (int d = 0; d < DIGITS; d++) {
    int8_t h1[18], s1[18], hw[10], sw[10];
    long t0 = usec();
    npu_fc(r, &mlp[0], &digit_px[d * 144], h1);
    npu_fc(r, &mlp[1], h1, hw);
    long t1 = usec();
    soft_fc(&mlp[0], &digit_px[d * 144], s1);
    soft_fc(&mlp[1], s1, sw);
    long t2 = usec();
    hw_us += t1 - t0;
    sw_us += t2 - t1;

    int same = 1, top = 0;
    for (int i = 0; i < 10; i++) {
      same &= hw[i] == sw[i] && hw[i] == digit_want[d * 10 + i];
      if (hw[i] > hw[top]) top = i;
    }
    printf("digit %d -> %d%s\n", digit_label[d], top, same && top == digit_label[d] ? "" : " BAD");
    good += same && top == digit_label[d];
  }
  printf("engine %ld us, software %ld us per image\n", hw_us / DIGITS, sw_us / DIGITS);
  printf("mnist npu %s %d/%d\n", good == DIGITS ? "ok" : "BAD", good, DIGITS);
  return good != DIGITS;
}
