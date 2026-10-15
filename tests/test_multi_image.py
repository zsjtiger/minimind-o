"""多图样本：<image> 标记与图像的绑定、collate 的补零堆叠。"""
import torch

from baize_v.data.collate import omni_collate_fn
from baize_v.data.dataset import OmniDataset


def _bare_dataset(max_images=4):
    ds = OmniDataset.__new__(OmniDataset)   # 只测试纯方法，不加载数据与处理器
    ds.image_token = '<PAD>' * 2
    ds.max_images = max_images
    return ds


def test_tags_bind_to_images_in_order():
    ds = _bare_dataset()
    px = torch.rand(2, 3, 8, 8)
    prompt, out = ds._bind_images('A<image>B<image>C', px)
    assert prompt == 'A<PAD><PAD>B<PAD><PAD>C'
    assert out.shape[0] == 2 and torch.equal(out, px)


def test_extra_tags_are_removed_and_extra_images_dropped():
    ds = _bare_dataset()
    prompt, out = ds._bind_images('<image><image><image>q', torch.rand(1, 3, 8, 8))
    assert prompt == '<PAD><PAD>q'
    assert out.shape[0] == 1


def test_prompt_without_tags_keeps_pixels():
    ds = _bare_dataset()
    px = torch.rand(1, 3, 8, 8)
    prompt, out = ds._bind_images('no image here', px)
    assert prompt == 'no image here' and out is px


def test_collate_pads_to_common_image_count():
    def sample(n):
        return (torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), torch.zeros(9, 2, dtype=torch.long),
                None, torch.tensor(0), torch.rand(n, 3, 8, 8), torch.zeros(1))
    *_, pixel_values, _ = omni_collate_fn([sample(2), sample(1)])
    assert pixel_values.shape == (2, 2, 3, 8, 8)
    assert pixel_values[1, 1].abs().sum() == 0   # 补的零图
    assert pixel_values[0, 1].abs().sum() > 0
