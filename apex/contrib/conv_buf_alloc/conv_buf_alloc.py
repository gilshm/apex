import pdb

import torch
from torch.autograd import gradcheck

from apex import check_cudnn_version_and_warn
import conv_buf_alloc

check_cudnn_version_and_warn(__name__, 8400)


class ConvBufAlloc_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, dx_buf, dw_buf, padding=0, stride=1, skip_wgrad=False):
        outputs = conv_buf_alloc.forward([x, weight], padding, stride)
        ctx.save_for_backward(x, weight, dx_buf, dw_buf)
        ctx.padding = padding
        ctx.stride = stride
        ctx.skip_wgrad = skip_wgrad

        return outputs[0]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        x, weight, dx_buf, dw_buf = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride
        skip_wgrad = ctx.skip_wgrad
        grads = conv_buf_alloc.backward([x, weight, grad_output],
                                        [dx_buf, dw_buf],
                                        padding, stride,
                                        skip_wgrad)

        return grads[0], grads[1], None, None, None, None, None, None


ConvBufAlloc = ConvBufAlloc_.apply

