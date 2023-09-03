#include <iostream>
#include <torch/torch.h>
#include <Libtorch_CUDA_helper.h>

int main()
{
    torch::Tensor t = torch::rand({1,1,1}).to(torch::kCUDA);
    auto result = tensor_batch_add_cuda(t,t);
    return 0;
}