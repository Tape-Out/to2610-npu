/* 到 to2610-npu 引擎寄存器的一条路。寄存器表照 hwsrc/npu_mmio.v 的文件头，偏移都是字节。 */
#ifndef TNPU_BUS_H
#define TNPU_BUS_H

#include <stddef.h>
#include <stdint.h>

typedef struct bus bus;

/* dev:/dev/npu       在芯片的 Linux 里，经驱动 mmap 寄存器那一页；
   tcp:主机:端口      另一头是测试台：一帧是长度（4 字节，大端）加 'w'/'r'、偏移、一个字（大端），回同样长的一帧 */
bus *bus_open (const char *spec);
void bus_close (bus *b);
int bus_wr (bus *b, unsigned off, uint32_t v);
int bus_rd (bus *b, unsigned off, uint32_t *v);

#endif
