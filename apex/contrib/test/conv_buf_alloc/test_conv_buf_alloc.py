import copy
import math
import random
import unittest

import torch
import torch.nn.functional as F

HAS_CONV_BUF_ALLOC = None
try:
    from apex.contrib.conv_buf_alloc import ConvBufAlloc
except ImportError as e:
    HAS_CONV_BUF_ALLOC = False
else:
    HAS_CONV_BUF_ALLOC = True


@unittest.skipIf(not HAS_CONV_BUF_ALLOC, "`apex.contrib.conv_buf_alloc` is not found.")
class FusedDenseTest(unittest.TestCase):
    def setUp(self, seed=0):
        super().setUp()
        torch.manual_seed(seed)

        self.batch_size = random.randint(1, 64)
        self.in_channels = random.randint(1, 64) * 8
        self.out_channels = random.randint(1, 64) * 8
        self.in_height = self.in_width = random.randint(5, 100)
        self.conv_kernel_size = random.randint(1, 5)
        self.conv_pad = random.randint(0, int(self.conv_kernel_size / 2))
        self.conv_stride = random.randint(1, 5)
        self.conv_dilation = 1
        self.out_height = self.out_width = \
            math.floor((self.in_height + 2 * self.conv_pad - \
                        self.conv_dilation * (self.conv_kernel_size - 1) - 1) / self.conv_stride + 1)

        self.x = torch.randint(low=-16, high=16,
                               size=[self.batch_size, self.in_channels, self.in_height, self.in_width]) \
                               .cuda().to(memory_format=torch.channels_last).float()
        self.x_ = self.x.clone()
        self.x.requires_grad_()
        self.x_.requires_grad_()

        self.mask = torch.randn([self.batch_size, self.out_channels, self.out_height, self.out_width]).cuda().to(memory_format=torch.channels_last)
        self.mask = (self.mask > 0).to(torch.int8)
        self.mask_ = self.mask.clone()

        self.conv1 = torch.nn.Conv2d(self.in_channels, self.out_channels, self.conv_kernel_size,
                                     stride=self.conv_stride, padding=self.conv_pad, bias=False).cuda().to(memory_format=torch.channels_last)
        self.conv1_ = copy.deepcopy(self.conv1)

        self.conv1_dx_buf = torch.zeros_like(self.x).half()
        self.conv1_dw_buf = torch.zeros_like(self.conv1.weight).half()

        print()
        print('> input=[{}, {}, {}, {}]'.format(self.batch_size, self.in_channels, self.in_height, self.in_width))
        print('> kernel=[{}, {}, {}, {}], stride={}, pad={}'.format(self.out_channels, self.in_channels,
                                                                    self.conv_kernel_size, self.conv_kernel_size,
								    self.conv_stride, self.conv_pad))

    def test_conv_buf_alloc(self):
        with torch.cuda.amp.autocast(dtype=torch.half):
            out = ConvBufAlloc(self.x, self.conv1.weight,
                               self.conv1_dx_buf, self.conv1_dw_buf,
                               self.conv_pad, self.conv_stride, False)
            loss = (out.float()**2).sum() / out.numel()
        loss.backward()

        with torch.cuda.amp.autocast(dtype=torch.half):
            out_ = self.conv1_(self.x_)
            loss_ = (out_.float()**2).sum() / out_.numel()
        loss_.backward()

        self.assertTrue(torch.allclose(out_, out, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.conv1_.weight.grad, self.conv1.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.x_.grad, self.x.grad, atol=1e-3, rtol=1e-3, equal_nan=True))


if __name__ == '__main__':
    unittest.main()

