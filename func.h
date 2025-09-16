#ifndef FUNC_N
#define FUNC_N


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

__global__ void sigmoid_gpu_backward(float* in_grad, float* out_grad, float* out, int size){
    CUDA_KERNAL_LOOP(i, size){
        float sig = 1 / (1 + exp(-out[i]));
        in_grad[i] = sig * (1 - sig) * out_grad[i];
    }
}

#endif