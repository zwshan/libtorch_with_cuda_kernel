#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>



template <typename T>
__global__ void tensor_batch_add(const int nthread,const T* input1,const T* input2,T* output) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthread; i = i + 1){
        output[i] = input1[i] + input2[i];
      }
}


std::tuple<at::Tensor, at::Tensor> tensor_batch_add_cuda(const at::Tensor& input1,const at::Tensor& input2) {
    AT_ASSERTM(input1.device().is_cuda(), "input1 must be a CUDA tensor");
    AT_ASSERTM(input2.device().is_cuda(), "input2 must be a CUDA tensor");

    //at::TensorArg 变量通常在函数中用于参数验证和错误报告
    at::TensorArg input1_t{ input1, "input1", 1 }, input2_t{ input2, "input2", 2 };

    //判断报错信息来自哪里
    at::CheckedFrom c = "libtorch cuda code";
    at::checkAllSameGPU(c, { input1_t, input2_t });
    at::checkAllSameType(c, { input1_t, input2_t });

    at::cuda::CUDAGuard device_guard(input1.device());
    at::Tensor output = at::zeros({ input1.size(0), input1.size(1), input1.size(2)}, input1.options());
    auto output_size = input1.size(0);


    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(std::min(at::cuda::ATenCeilDiv(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),static_cast<int64_t>(4096)));
    dim3 block(512);

    //在前面创建了 output 的 zero 数组，所以这里它的元素总数不会为 0，否则就会报错
    if (output.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(output,output);
    }

    //&的意思是匿名函数内可以使用匿名函数外面的所有变量
    //通过AT_DISPATCH_FLOATING_TYPES_AND_HALF这个宏定义来调用 cuda 代码
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "libtorch cuda test", [&] {
        tensor_batch_add<scalar_t> <<<grid, block, 0, stream >>> (
            output_size,
            input1.contiguous().data_ptr<scalar_t>(),
            input2.contiguous().data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
        });
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output,output);
}
