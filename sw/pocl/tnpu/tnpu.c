/* to2610-npu 的乘加引擎当作 PoCL 的一个设备，认一个内建内核 pocl.dnn.dense.relu.i8。
   PoCL 只定了这个内核的名字与参数表（input、weights、output、bias、scale、shift、zero_point、output_minus、input_size），
   没定算法；这里照 int8 的全连接定：
     acc[j] = bias[j] + Σk input[k]·weights[j·input_size + k]，ReLU，
     output[j] = clamp(((acc[j]·scale) >> shift) + zero_point, −8, 7)
   全局大小是输出个数，可以带全局偏移（每个输出各用一组 scale、shift 时一个输出起跑一次）。
   引擎只有 int4 的乘加与 16 位的累加器：输入、权、零点不在 −8..7，偏置出了 int16，scale 出了 15 位，shift 出了 5 位，
   output_minus 不是 0，都报错，不截断。算法与 sw/npu/npu.h 的 npu_fc 一步不差，寄存器照 hwsrc/npu_mmio.v。 */

#include "tnpu.h"

#include <string.h>

#include "bus.h"
#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl_builtin_kernels.h"
#include "pocl_mem_management.h"
#include "pocl_util.h"
#include "utlist.h"

enum
{
  DAT = 0x00,
  ACC = 0x04,
  OUT = 0x08,
  SHAMT = 0x0C,
  QMUL = 0x10,
  CTRL = 0x14,
  ID = 0x1C
};
#define TNPU_ID 0x4E505531u

typedef struct
{
  bus *b;
  cl_bool available;
  pocl_lock_t cq_lock;
  _cl_command_node *ready_list;
  _cl_command_node *command_list;
} tnpu_data;

static int
int4 (int v)
{
  return v >= -8 && v <= 7;
}

static const char *
dense (tnpu_data *d, const int8_t *in, const int8_t *w, int8_t *out,
       const int32_t *bias, uint32_t scale, uint32_t shift, int32_t zp,
       uint32_t k, size_t off, size_t n)
{
  if (!int4 (zp) || scale >> 15 || shift >> 5)
    return "tnpu：零点要在 −8..7，scale 在 15 位、shift 在 5 位以内";
  for (uint32_t i = 0; i < k; ++i)
    if (!int4 (in[i]))
      return "tnpu：输入要在 −8..7（引擎是 int4）";
  if (bus_wr (d->b, CTRL, (uint32_t)(zp & 15) << 1 | 1))
    return "tnpu：寄存器写不进去";
  for (size_t j = off; j < off + n; ++j)
    {
      if (bias[j] < -32768 || bias[j] > 32767)
        return "tnpu：偏置要在 int16 以内";
      if (bus_wr (d->b, SHAMT, shift) || bus_wr (d->b, QMUL, scale)
          || bus_wr (d->b, ACC, (uint16_t)bias[j]))
        return "tnpu：寄存器写不进去";
      const int8_t *wj = w + j * k;
      for (uint32_t i = 0; i < k; i += 4)
        {
          uint32_t v = 0;
          for (uint32_t t = 0; t < 4 && i + t < k; ++t)
            {
              if (!int4 (wj[i + t]))
                return "tnpu：权要在 −8..7（引擎是 int4）";
              v |= (uint32_t)(in[i + t] & 15) << (4 * t)
                   | (uint32_t)(wj[i + t] & 15) << (16 + 4 * t);
            }
          if (bus_wr (d->b, DAT, v))
            return "tnpu：寄存器写不进去";
        }
      uint32_t r;
      if (bus_rd (d->b, OUT, &r))
        return "tnpu：寄存器读不回来";
      out[j] = (int8_t)r;
    }
  return NULL;
}

static const char *
run (_cl_command_node *node)
{
  tnpu_data *d = node->device->data;
  pocl_kernel_metadata_t *meta = node->command.run.kernel->meta;
  struct pocl_context *pc = &node->command.run.pc;
  if (meta->builtin_kernel_id != POCL_CDBI_DNN_DENSE_RELU_I8 || meta->num_args != 9)
    return "tnpu：只认 pocl.dnn.dense.relu.i8";
  if (pc->num_groups[1] * pc->local_size[1] != 1 || pc->num_groups[2] * pc->local_size[2] != 1)
    return "tnpu：只有一维";
  size_t n = pc->num_groups[0] * pc->local_size[0], off = pc->global_offset[0];
  if (n == 0)
    return NULL;

  void *buf[4];
  size_t size[4];
  uint32_t pod[5];
  for (unsigned i = 0; i < 9; ++i)
    {
      struct pocl_argument *al = &node->command.run.arguments[i];
      if (!al->value)
        return "tnpu：参数没设";
      if (i < 4)
        {
          if (meta->arg_info[i].type != POCL_ARG_TYPE_POINTER || al->is_raw_ptr)
            return "tnpu：前四个参数要是缓冲区";
          cl_mem m = *(cl_mem *)al->value;
          buf[i] = m->device_ptrs[node->device->global_mem_id].mem_ptr;
          size[i] = m->size;
        }
      else
        memcpy (&pod[i - 4], al->value, sizeof pod[0]);
    }
  uint32_t k = pod[4];
  if (pod[3])
    return "tnpu：output_minus 要是 0";
  if (k == 0 || size[0] < k || size[1] < (off + n) * k || size[2] < off + n
      || size[3] < (off + n) * sizeof (int32_t))
    return "tnpu：缓冲区比 input_size 与全局大小要的小";
  return dense (d, buf[0], buf[1], buf[2], buf[3], pod[0], pod[1], (int32_t)pod[2], k, off, n);
}

static void
exec (_cl_command_node *node)
{
  if (node->type != CL_COMMAND_NDRANGE_KERNEL)
    {
      pocl_exec_command (node);
      return;
    }
  cl_event ev = node->sync.event.event;
  pocl_update_event_running (ev);
  const char *err = run (node);
  if (err)
    {
      POCL_MSG_ERR ("%s\n", err);
      POCL_UPDATE_EVENT_FAILED_MSG (CL_FAILED, ev, "tnpu NDRange          ");
    }
  else
    POCL_UPDATE_EVENT_COMPLETE_MSG (ev, "tnpu NDRange          ");
}

/* 命令的收发照 cpu-minimal：引擎一次只算一层，就绪一条执行一条 */
static void
schedule (tnpu_data *d)
{
  _cl_command_node *node;
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready (node->sync.event.event));
      assert (node->sync.event.event->status == CL_SUBMITTED);
      CDL_DELETE (d->ready_list, node);
      POCL_UNLOCK (d->cq_lock);
      exec (node);
      POCL_LOCK (d->cq_lock);
    }
}

static void
tnpu_submit (_cl_command_node *node, cl_command_queue cq)
{
  tnpu_data *d = node->device->data;
  node->state = POCL_COMMAND_READY;
  POCL_LOCK (d->cq_lock);
  pocl_command_push (node, &d->ready_list, &d->command_list);
  POCL_UNLOCK_OBJ (node->sync.event.event);
  schedule (d);
  POCL_UNLOCK (d->cq_lock);
}

static void
tnpu_join (cl_device_id device, cl_command_queue cq)
{
  tnpu_data *d = device->data;
  POCL_LOCK (d->cq_lock);
  schedule (d);
  POCL_UNLOCK (d->cq_lock);
}

static void
tnpu_notify (cl_device_id device, cl_event event, cl_event finished)
{
  tnpu_data *d = device->data;
  _cl_command_node *volatile node = event->command;
  if (finished->status < CL_COMPLETE)
    {
      pocl_unlock_events_inorder (event, finished);
      pocl_update_event_failed (CL_FAILED, NULL, 0, event, NULL);
      pocl_lock_events_inorder (finished, event);
      return;
    }
  if (node->state != POCL_COMMAND_READY || !pocl_command_is_ready (event) || event->status != CL_QUEUED)
    return;
  pocl_update_event_submitted (event);
  POCL_LOCK (d->cq_lock);
  CDL_DELETE (d->command_list, node);
  CDL_PREPEND (d->ready_list, node);
  POCL_UNLOCK_OBJ (event);
  schedule (d);
  POCL_LOCK_OBJ (event);
  POCL_UNLOCK (d->cq_lock);
}

static int
tnpu_supports_binary (cl_device_id device, size_t length, const char *binary)
{
  return 0;
}

static int
tnpu_build_binary (cl_program program, cl_uint device_i, int link_program, int spir_build)
{
  return CL_INVALID_BINARY;
}

static int
tnpu_build_builtin (cl_program program, cl_uint device_i)
{
  return CL_SUCCESS;
}

static int
tnpu_setup_metadata (cl_device_id device, cl_program program, unsigned program_device_i)
{
  return program->builtin_kernel_names ? pocl_setup_builtin_metadata (device, program, program_device_i) : 0;
}

static void
tnpu_local_size (cl_device_id dev, cl_kernel kernel, unsigned device_i, size_t max_group_size,
                 size_t gx, size_t gy, size_t gz, size_t *lx, size_t *ly, size_t *lz)
{
  *lx = *ly = *lz = 1;
}

static char *
tnpu_build_hash (cl_device_id device)
{
  return strdup ("tnpu");
}

static unsigned
tnpu_probe (struct pocl_device_ops *ops)
{
  int n = pocl_device_get_env_count (ops->device_name);
  return n < 0 ? 0 : n;
}

static cl_int
tnpu_init (unsigned j, cl_device_id dev, const char *parameters)
{
  if (!parameters)
    {
      POCL_MSG_ERR ("tnpu：POCL_TNPU%u_PARAMETERS 要给寄存器的路，dev:/dev/npu 或 tcp:主机:端口\n", j);
      return CL_INVALID_DEVICE;
    }
  tnpu_data *d = calloc (1, sizeof *d);
  if (!d)
    return CL_OUT_OF_HOST_MEMORY;
  uint32_t id = 0;
  if (!(d->b = bus_open (parameters)) || bus_rd (d->b, ID, &id) || id != TNPU_ID)
    {
      POCL_MSG_ERR ("tnpu：%s 上的标识读出 0x%08x，不是 to2610-npu 的引擎\n", parameters, id);
      bus_close (d->b);
      free (d);
      return CL_INVALID_DEVICE;
    }
  pocl_init_default_device_infos (dev, "");
  SETUP_DEVICE_CL_VERSION (dev, 1, 2);
  dev->type = CL_DEVICE_TYPE_ACCELERATOR;
  dev->long_name = "to2610-npu";
  dev->short_name = "tnpu";
  dev->vendor = "Tape-Out";
  dev->profile = "EMBEDDED_PROFILE";
  dev->max_compute_units = 1;
  dev->max_work_group_size = dev->max_work_item_sizes[0] = 1;
  dev->max_work_item_sizes[1] = dev->max_work_item_sizes[2] = 1;
  dev->preferred_wg_size_multiple = 1;
  dev->global_mem_size = dev->max_mem_alloc_size = 1 << 20;
  dev->image_support = CL_FALSE;
  dev->compiler_available = dev->linker_available = CL_FALSE;
  dev->execution_capabilities = CL_EXEC_KERNEL;
  dev->builtin_kernel_list = strdup ("pocl.dnn.dense.relu.i8");
  dev->num_builtin_kernels = 1;
  pocl_setup_builtin_kernels_with_version (dev);
  d->available = CL_TRUE;
  dev->available = &d->available;
  dev->data = d;
  POCL_INIT_LOCK (d->cq_lock);
  return CL_SUCCESS;
}

static cl_int
tnpu_uninit (unsigned j, cl_device_id dev)
{
  tnpu_data *d = dev->data;
  POCL_DESTROY_LOCK (d->cq_lock);
  bus_close (d->b);
  free (d);
  dev->data = NULL;
  return CL_SUCCESS;
}

void
pocl_tnpu_init_device_ops (struct pocl_device_ops *ops)
{
  ops->device_name = "tnpu";
  ops->probe = tnpu_probe;
  ops->init = tnpu_init;
  ops->uninit = tnpu_uninit;
  ops->build_hash = tnpu_build_hash;

  ops->alloc_mem_obj = pocl_driver_alloc_mem_obj;
  ops->free = pocl_driver_free;
  ops->read = pocl_driver_read;
  ops->read_rect = pocl_driver_read_rect;
  ops->write = pocl_driver_write;
  ops->write_rect = pocl_driver_write_rect;
  ops->copy = pocl_driver_copy;
  ops->copy_with_size = pocl_driver_copy_with_size;
  ops->copy_rect = pocl_driver_copy_rect;
  ops->memfill = pocl_driver_memfill;
  ops->map_mem = pocl_driver_map_mem;
  ops->unmap_mem = pocl_driver_unmap_mem;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->supports_binary = tnpu_supports_binary;
  ops->build_binary = tnpu_build_binary;
  ops->build_builtin = tnpu_build_builtin;
  ops->setup_metadata = tnpu_setup_metadata;
  ops->free_program = pocl_driver_free_program;
  ops->compute_local_size = tnpu_local_size;

  ops->submit = tnpu_submit;
  ops->join = tnpu_join;
  ops->flush = tnpu_join;
  ops->notify = tnpu_notify;
  ops->broadcast = pocl_broadcast;
}
