/* 经 OpenCL 用 to2610-npu：只调标准的 OpenCL 1.2 接口与 PoCL 的内建内核 pocl.dnn.dense.relu.i8。
     nnrun info          平台与设备
     nnrun mnist         随片的 MNIST 模型两层都走这个内核，每个输出一组乘数与移位，起跑一次；
                         与 sw/npu/npu.h 的 soft_fc 逐项比（第二层也带 ReLU：PoCL 只有带 ReLU 的这一个），分类与标签比
     nnrun random [层数]  随机的层，整层一组乘数与移位，一次起跑，与 soft_fc 逐项比
   设备由 PoCL 的环境变量选：POCL_DEVICES=tnpu POCL_TNPU0_PARAMETERS=tcp:127.0.0.1:2611（或 dev:/dev/npu） */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npu.h"
#include "model.h"

#define CHECK(x)                                                              \
  do                                                                          \
    {                                                                         \
      cl_int e_ = (x);                                                        \
      if (e_ != CL_SUCCESS)                                                   \
        {                                                                     \
          fprintf (stderr, "%s：%d\n", #x, e_);                              \
          exit (1);                                                           \
        }                                                                     \
    }                                                                         \
  while (0)

static cl_device_id dev;
static cl_context ctx;
static cl_command_queue q;
static cl_kernel kr;

static void
open_device (void)
{
  cl_platform_id plat;
  CHECK (clGetPlatformIDs (1, &plat, NULL));
  CHECK (clGetDeviceIDs (plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL));
  cl_int e;
  ctx = clCreateContext (NULL, 1, &dev, NULL, NULL, &e);
  CHECK (e);
  q = clCreateCommandQueue (ctx, dev, 0, &e);
  CHECK (e);
  cl_program p = clCreateProgramWithBuiltInKernels (ctx, 1, &dev, "pocl.dnn.dense.relu.i8", &e);
  CHECK (e);
  CHECK (clBuildProgram (p, 1, &dev, NULL, NULL, NULL));
  kr = clCreateKernel (p, "pocl.dnn.dense.relu.i8", &e);
  CHECK (e);
}

static cl_mem
buffer (size_t n, const void *init)
{
  cl_int e;
  cl_mem m = clCreateBuffer (ctx, CL_MEM_READ_WRITE | (init ? CL_MEM_COPY_HOST_PTR : 0), n, (void *)init, &e);
  CHECK (e);
  return m;
}

/* 一层：权按 [输入][输出] 来（同 struct fc），内核要 [输出][输入]。each 为真时每个输出单独起跑，用它自己的乘数与移位 */
static void
layer (const struct fc *l, const int8_t *in, int8_t *out, int each)
{
  int8_t *wt = malloc ((size_t)(l->k * l->n));
  int32_t *b = malloc (sizeof *b * (size_t)l->n);
  for (int j = 0; j < l->n; ++j)
    {
      b[j] = l->bias[j];
      for (int i = 0; i < l->k; ++i)
        wt[j * l->k + i] = l->w[i * l->n + j];
    }
  cl_mem mi = buffer ((size_t)l->k, in), mw = buffer ((size_t)(l->k * l->n), wt), mo = buffer ((size_t)l->n, NULL),
         mb = buffer (sizeof *b * (size_t)l->n, b);
  cl_uint zp = (cl_uint)l->zp, minus = 0, k = (cl_uint)l->k;
  CHECK (clSetKernelArg (kr, 0, sizeof mi, &mi));
  CHECK (clSetKernelArg (kr, 1, sizeof mw, &mw));
  CHECK (clSetKernelArg (kr, 2, sizeof mo, &mo));
  CHECK (clSetKernelArg (kr, 3, sizeof mb, &mb));
  CHECK (clSetKernelArg (kr, 6, sizeof zp, &zp));
  CHECK (clSetKernelArg (kr, 7, sizeof minus, &minus));
  CHECK (clSetKernelArg (kr, 8, sizeof k, &k));
  for (size_t j = 0, one = 1, all = (size_t)l->n; j < (each ? all : 1); ++j)
    {
      cl_uint scale = l->qmul[j], shift = l->shamt[j];
      CHECK (clSetKernelArg (kr, 4, sizeof scale, &scale));
      CHECK (clSetKernelArg (kr, 5, sizeof shift, &shift));
      CHECK (clEnqueueNDRangeKernel (q, kr, 1, each ? &j : NULL, each ? &one : &all, NULL, 0, NULL, NULL));
    }
  CHECK (clEnqueueReadBuffer (q, mo, CL_TRUE, 0, (size_t)l->n, out, 0, NULL, NULL));
  clReleaseMemObject (mi);
  clReleaseMemObject (mw);
  clReleaseMemObject (mo);
  clReleaseMemObject (mb);
  free (wt);
  free (b);
}

static int
info (void)
{
  char name[256], vendor[256], version[256], bik[1024];
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_NAME, sizeof name, name, NULL));
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_VENDOR, sizeof vendor, vendor, NULL));
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_VERSION, sizeof version, version, NULL));
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_BUILT_IN_KERNELS, sizeof bik, bik, NULL));
  printf ("%s（%s）%s\n内建内核 %s\n", name, vendor, version, bik);
  return 0;
}

static int
mnist (void)
{
  struct fc l2 = mlp[1];
  l2.relu = 1;
  int wrong = 0, miss = 0;
  for (int d = 0; d < DIGITS; ++d)
    {
      const int8_t *x = digit_px + d * mlp[0].k;
      int8_t h[64], o[16], hs[64], os[16];
      layer (&mlp[0], x, h, 1);
      layer (&l2, h, o, 1);
      soft_fc (&mlp[0], x, hs);
      soft_fc (&l2, hs, os);
      int best = 0;
      for (int i = 0; i < l2.n; ++i)
        {
          wrong += o[i] != os[i];
          best = o[i] > o[best] ? i : best;
        }
      for (int i = 0; i < mlp[0].n; ++i)
        wrong += h[i] != hs[i];
      miss += best != digit_label[d];
      printf ("%d", best);
    }
  printf ("\nMNIST %d 张：与 soft_fc 不同 %d 格，分类错 %d 张\n", DIGITS, wrong, miss);
  return wrong || miss;
}

static int
random_layers (int count)
{
  srand (2611);
  int bad = 0;
  for (int c = 0; c < count; ++c)
    {
      struct fc l = { .k = 1 + rand () % 144, .n = 1 + rand () % 18, .relu = 1, .zp = rand () % 16 - 8 };
      int8_t *w = malloc ((size_t)(l.k * l.n)), in[144], out[18], want[18];
      int16_t b[18];
      uint16_t qm[18];
      uint8_t sh[18];
      for (int i = 0; i < l.k * l.n; ++i)
        w[i] = (int8_t)(rand () % 16 - 8);
      for (int i = 0; i < l.k; ++i)
        in[i] = (int8_t)(rand () % 16 - 8);
      qm[0] = (uint16_t)(1 + rand () % 32767);
      sh[0] = (uint8_t)(8 + rand () % 16);
      for (int j = 0; j < l.n; ++j)
        b[j] = (int16_t)(rand () % 2001 - 1000), qm[j] = qm[0], sh[j] = sh[0];
      l.w = w, l.bias = b, l.qmul = qm, l.shamt = sh;
      layer (&l, in, out, 0);
      soft_fc (&l, in, want);
      bad += memcmp (out, want, (size_t)l.n) != 0;
      free (w);
    }
  printf ("随机 %d 层：与 soft_fc 不同 %d 层\n", count, bad);
  return bad != 0;
}

int
main (int argc, char **argv)
{
  if (argc < 2)
    {
      fprintf (stderr, "用法：nnrun info | mnist | random [层数]\n");
      return 2;
    }
  open_device ();
  if (!strcmp (argv[1], "info"))
    return info ();
  if (!strcmp (argv[1], "mnist"))
    return mnist ();
  if (!strcmp (argv[1], "random"))
    return random_layers (argc > 2 ? atoi (argv[2]) : 20);
  fprintf (stderr, "不认识的命令 %s\n", argv[1]);
  return 2;
}
