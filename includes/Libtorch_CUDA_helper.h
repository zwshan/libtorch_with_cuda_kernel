#pragma once
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>



std::tuple<at::Tensor, at::Tensor> tensor_batch_add_cuda(
    const at::Tensor& input1,
    const at::Tensor& input2);

