#include "bus.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#define SPAN 0x20

struct bus
{
  int fd;
  volatile uint32_t *regs;
};

static int
dial (const char *hostport)
{
  const char *colon = strrchr (hostport, ':');
  char host[256];
  if (!colon || (size_t)(colon - hostport) >= sizeof host)
    return -1;
  memcpy (host, hostport, colon - hostport);
  host[colon - hostport] = 0;
  struct addrinfo hints = { .ai_socktype = SOCK_STREAM }, *res;
  if (getaddrinfo (host, colon + 1, &hints, &res))
    return -1;
  int fd = -1;
  for (struct addrinfo *ai = res; ai && fd < 0; ai = ai->ai_next)
    {
      fd = socket (ai->ai_family, ai->ai_socktype, ai->ai_protocol);
      if (fd >= 0 && connect (fd, ai->ai_addr, ai->ai_addrlen))
        {
          close (fd);
          fd = -1;
        }
    }
  freeaddrinfo (res);
  /* 一次寄存器访问一个来回，Nagle 会让每一来回多等一个 ACK */
  int one = 1;
  if (fd >= 0)
    setsockopt (fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  return fd;
}

bus *
bus_open (const char *spec)
{
  bus *b = calloc (1, sizeof *b);
  if (!b)
    return NULL;
  b->fd = -1;
  if (!strncmp (spec, "tcp:", 4))
    b->fd = dial (spec + 4);
  else if (!strncmp (spec, "dev:", 4) && (b->fd = open (spec + 4, O_RDWR | O_SYNC)) >= 0)
    {
      void *p = mmap (NULL, SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, b->fd, 0);
      b->regs = p == MAP_FAILED ? NULL : p;
      if (!b->regs)
        {
          close (b->fd);
          b->fd = -1;
        }
    }
  if (b->fd < 0)
    {
      free (b);
      return NULL;
    }
  return b;
}

void
bus_close (bus *b)
{
  if (!b)
    return;
  if (b->regs)
    munmap ((void *)b->regs, SPAN);
  close (b->fd);
  free (b);
}

static int
whole (int fd, void *buf, size_t n, int out)
{
  for (uint8_t *p = buf; n;)
    {
      ssize_t k = out ? write (fd, p, n) : read (fd, p, n);
      if (k < 0 && errno == EINTR)
        continue;
      if (k <= 0)
        return -1;
      p += k;
      n -= k;
    }
  return 0;
}

static int
xfer (bus *b, char op, unsigned off, uint32_t *v)
{
  uint8_t f[10] = { 0, 0, 0, 6, (uint8_t)op, (uint8_t)off, *v >> 24, *v >> 16, *v >> 8, *v };
  if (whole (b->fd, f, sizeof f, 1) || whole (b->fd, f, sizeof f, 0))
    return -1;
  *v = (uint32_t)f[6] << 24 | f[7] << 16 | f[8] << 8 | f[9];
  return 0;
}

int
bus_wr (bus *b, unsigned off, uint32_t v)
{
  if (b->regs)
    {
      b->regs[off / 4] = v;
      return 0;
    }
  return xfer (b, 'w', off, &v);
}

int
bus_rd (bus *b, unsigned off, uint32_t *v)
{
  if (b->regs)
    {
      *v = b->regs[off / 4];
      return 0;
    }
  *v = 0;
  return xfer (b, 'r', off, v);
}
