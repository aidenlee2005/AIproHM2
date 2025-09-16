#include "tensor.h"

#define CUDA_KERNAL_LOOP(i,n)\
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)

__global__ void relu_gpu(float* in, float* out, int size){
    CUDA_KERNAL_LOOP(i, size){
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

__global__ void relu_gpu_backward(float* in_grad, float* out_grad, float* x, int size){
    CUDA_KERNAL_LOOP(i, size){
        in_grad[i] = x[i] > 0 ? out_grad[i] : 0;
    }
}

__global__ void sigmoid_gpu(float* in, float* out, int size){
    CUDA_KERNAL_LOOP(i, size){
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

__global__ void sigmoid_gpu_backward(float* in_grad, float* out_grad, float* x, int size){
    CUDA_KERNAL_LOOP(i, size){
        float sig = 1 / (1 + exp(-x[i]));
        in_grad[i] = sig * (1 - sig) * out_grad[i];
    }
}

int main(){

    // Test1 Tensor类的实现
    std::cout << "Test1 Tensor类的实现" << std::endl;
    Tensor t({3,2}, Device::CPU);
    float* data = t.get_data();
    for (int i = 0; i < t.get_size(); ++i){
        data[i] = i - 3;
    }
    t.print();

    Tensor g({2,3},Device::GPU);
    float* g_data = (float*) malloc(g.get_size());
    cudaMemcpy(g_data, g.get_data(), g.get_size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < g.get_size(); ++i){
        g_data[i] = i*2 - 6;
    }
    cudaMemcpy(g.get_data(), g_data, g.get_size() * sizeof(float), cudaMemcpyHostToDevice);
    g.print();

    free(g_data);

    Tensor f = t.gpu();
    t.print();
    Tensor h = g.cpu();
    h.print();


    // Test2 relu和sigmoid函数的实现
    std::cout << "Test2 relu函数的实现" << std::endl;
    relu_gpu<<<1, g.get_size()>>>(g.get_data(), g.get_data(), g.get_size());
    g.print();
    std::cout << "Test2 sigmoid函数的实现" << std::endl;
    sigmoid_gpu<<<1, g.get_size()>>>(g.get_data(), g.get_data(), g.get_size());
    g.print();

    // Test3 relu_backward函数的实现
    std::cout << "Test3 relu_backward函数的实现" << std::endl;
    Tensor out_grad({2,3}, Device::GPU);
    Tensor cpu_tmp = out_grad.cpu();
    for (int i = 0; i < out_grad.get_size(); ++i){
        cpu_tmp.get_data()[i] = i*2 - 3;  // -3, -1, 1, 3, 5, 7
    }
    out_grad = cpu_tmp.gpu();

    Tensor in_grad({2,3}, Device::GPU);

    Tensor x({2,3}, Device::GPU);
    Tensor x_cpu_tmp = x.cpu();
    for (int i = 0; i < x.get_size(); ++i){
        x_cpu_tmp.get_data()[i] = i - 3; // -3, -2, -1, 0, 1, 2
    }
    x = x_cpu_tmp.gpu();

    relu_gpu_backward<<<1, x.get_size()>>>(in_grad.get_data(), out_grad.get_data(), x.get_data(), x.get_size());
    in_grad.print();

    std::cout << "Test3 sigmoid_backward函数的实现" << std::endl;
    sigmoid_gpu_backward<<<1, x.get_size()>>>(in_grad.get_data(), out_grad.get_data(), x.get_data(), x.get_size());
    in_grad.print();

}
