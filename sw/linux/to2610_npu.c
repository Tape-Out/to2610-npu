// SPDX-License-Identifier: GPL-2.0
/*
 * The MLP engine on the to2610-npu chip.
 *
 * One register write makes the engine multiply and accumulate four int4 pairs, so the
 * useful interface is the register page itself: /dev/npu maps it and user space drives
 * the engine with plain stores. The driver finds the engine, checks that it answers and
 * hands out that one page.
 */
#include <linux/io.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>

#define NPU_ID		0x1c
#define NPU_MAGIC	0x4e505531

struct to2610_npu {
	struct miscdevice misc;
	resource_size_t phys;
};

static int to2610_npu_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct to2610_npu *npu = container_of(file->private_data, struct to2610_npu, misc);

	if (vma->vm_pgoff || vma->vm_end - vma->vm_start > PAGE_SIZE)
		return -EINVAL;

	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	return io_remap_pfn_range(vma, vma->vm_start, npu->phys >> PAGE_SHIFT,
				  vma->vm_end - vma->vm_start, vma->vm_page_prot);
}

static const struct file_operations to2610_npu_fops = {
	.owner = THIS_MODULE,
	.mmap = to2610_npu_mmap,
};

static int to2610_npu_probe(struct platform_device *pdev)
{
	struct to2610_npu *npu;
	struct resource *res;
	void __iomem *regs;
	u32 id;
	int ret;

	npu = devm_kzalloc(&pdev->dev, sizeof(*npu), GFP_KERNEL);
	if (!npu)
		return -ENOMEM;

	regs = devm_platform_get_and_ioremap_resource(pdev, 0, &res);
	if (IS_ERR(regs))
		return PTR_ERR(regs);

	/* The same image boots on to2610-kvc, where nothing answers at this address. */
	id = readl(regs + NPU_ID);
	if (id != NPU_MAGIC) {
		dev_info(&pdev->dev, "no engine here (id %08x)\n", id);
		return -ENODEV;
	}

	npu->phys = res->start;
	npu->misc.minor = MISC_DYNAMIC_MINOR;
	npu->misc.name = "npu";
	npu->misc.fops = &to2610_npu_fops;
	npu->misc.parent = &pdev->dev;
	platform_set_drvdata(pdev, npu);

	ret = misc_register(&npu->misc);
	if (ret)
		return ret;

	dev_info(&pdev->dev, "MLP engine at %pa\n", &res->start);
	return 0;
}

static void to2610_npu_remove(struct platform_device *pdev)
{
	struct to2610_npu *npu = platform_get_drvdata(pdev);

	misc_deregister(&npu->misc);
}

static const struct of_device_id to2610_npu_of_match[] = {
	{ .compatible = "tapeout,to2610-npu" },
	{ }
};
MODULE_DEVICE_TABLE(of, to2610_npu_of_match);

static struct platform_driver to2610_npu_driver = {
	.probe = to2610_npu_probe,
	.remove = to2610_npu_remove,
	.driver = {
		.name = "to2610-npu",
		.of_match_table = to2610_npu_of_match,
	},
};
module_platform_driver(to2610_npu_driver);

MODULE_DESCRIPTION("MLP engine of the to2610-npu chip");
MODULE_LICENSE("GPL");
