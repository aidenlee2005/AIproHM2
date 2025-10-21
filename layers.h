#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"
#include "func.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#ifndef CUDA_KERNAL_LOOP
#define CUDA_KERNAL_LOOP(i,n)\
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
#endif

enum class TransposeType {
    NoTranspose,
    Transpose,
};

__global__ void fill_elements(float* data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

//C(m,n) = alf * A(m,k) * B(k,n) + bet * C(m,n)
// void gemm_gpu(TransposeType transA, TransposeType transB, const float* A, const float* B, float* C, 
//     const int m, const int n, const int k, const float alf, const float bet){
//         int lda, ldb, ldc;
//         ldc = m;
//         const float* alpha = &alf;
//         const float* beta = &bet;
//         cublasHandle_t handle; cublasCreate(&handle);
//         lda = (transA == TransposeType::NoTranspose) ? m : k;
//         ldb = (transB == TransposeType::NoTranspose) ? k : n;
//         cublasOperation_t cuTransA = (transA == TransposeType::NoTranspose) ? CUBLAS_OP_N : CUBLAS_OP_T;
//         cublasOperation_t cuTransB = (transB == TransposeType::NoTranspose) ? CUBLAS_OP_N : CUBLAS_OP_T;
//         cublasSgemm(handle, cuTransA, cuTransB,
//              m, n, k, 
//              alpha, 
//              A, lda, 
//              B, ldb, 
//              beta, 
//              C, ldc);
//         cublasDestroy(handle);
//     }

void gemm_gpu(TransposeType transA, TransposeType transB,
              const float* A, const float* B, float* C,
              const int m, const int n, const int k,
              const float alf, const float bet, cudaStream_t stream = 0){

    // 语义（row-major）: C (m x n) = alf * op(A) (m x k) * op(B) (k x n) + bet * C (m x n)
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasCreate failed: " << stat << std::endl;
        return;
    }

    stat = cublasSetStream(handle, stream);
    if (stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasSetStream failed: " << stat << std::endl;
        cublasDestroy(handle);
        return;
    }

    const float* alpha = &alf;
    const float* beta  = &bet;

    // 将 row-major 映射到 cuBLAS(column-major)：计算 C_col = alpha * op_row(B)^T * op_row(A)^T + beta * C_col
    // 因此在 cublas 中传入 (B pointer) as A_arg, (A pointer) as B_arg，尺寸为 (n x m x k)
    cublasOperation_t opA_col = (transB == TransposeType::NoTranspose) ? CUBLAS_OP_N : CUBLAS_OP_T; // applied to B pointer (A_arg)
    cublasOperation_t opB_col = (transA == TransposeType::NoTranspose) ? CUBLAS_OP_N : CUBLAS_OP_T; // applied to A pointer (B_arg)

    // Leading dims: 当以 column-major 视角传入时，
    // - A_arg = B pointer 被视作列主序矩阵，其行数 = (transB == No) ? n : k
    // - B_arg = A pointer 被视作列主序矩阵，其行数 = (transA == No) ? k : m
    int lda = (transB == TransposeType::NoTranspose) ? n : k; // leading dim for B pointer (passed as A_arg)
    int ldb = (transA == TransposeType::NoTranspose) ? k : m; // leading dim for A pointer (passed as B_arg)
    int ldc = n; // C_col has rows = n

    stat = cublasSgemm(handle,
                       opA_col, opB_col,
                       /*m*/ n, /*n*/ m, /*k*/ k,
                       alpha,
                       B, lda,   // pass B as first operand to cublas
                       A, ldb,   // pass A as second operand
                       beta,
                       C, ldc);  // C is stored row-major, viewed as (n x m) in column-major
    if (stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasSgemm failed: " << stat << std::endl;
    }

    cublasDestroy(handle);
}

__global__ void quantize_one_decimal(float* data, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        data[idx] = roundf(data[idx] * 10.0f) / 10.0f;
    }
}

//random filling
void matrix_init(float* A, int n, unsigned long long seed = 123456){
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, A, n);
    curandDestroyGenerator(gen);

    int bs = 256;
    int gs = (n + bs - 1) / bs;
    quantize_one_decimal<<<gs, bs>>>(A, n);
}


//Task1: Fully Connected Layer
void forward_fc(float* input, float* output, float* weight, float* bias,
    int batch_size, int out_features, int in_features){
        //output(b, o) = input(b, i) * weight(i, o)
        gemm_gpu(TransposeType::NoTranspose,TransposeType::NoTranspose,
            input, weight, output, 
            batch_size, out_features, in_features, 
            1.0f, 0.0f);
        //output(b, o) += ones (b, 1) * bias(1, o)
        float* d_ones;
        cudaMalloc(&d_ones, batch_size * sizeof(float));
        cudaMemset(d_ones, 0, batch_size * sizeof(float));
        fill_elements<<<(batch_size + 255)/256, 256>>>(d_ones, batch_size, 1.0f);
        gemm_gpu(TransposeType::NoTranspose, TransposeType::NoTranspose,
            d_ones, bias, output, 
            batch_size, out_features, 1, 
            1.0f, 1.0f);
        cudaFree(d_ones);
    }

void backward_fc(float* input, float* weight, float* bias,
    int batch_size, int out_features, int in_features,
    float* grad_input, float* grad_output, float* grad_weight, float* grad_bias){
        // grad_input(b, i) = grad_output(b, o) * weight(i, o)^T
        gemm_gpu(TransposeType::NoTranspose, TransposeType::Transpose,
            grad_output, weight, grad_input, 
            batch_size, in_features, out_features, 
            1.0f, 0.0f);    
        // grad_weight(i, o) = input(b, i)^T * grad_output(b, o)
        gemm_gpu(TransposeType::Transpose, TransposeType::NoTranspose,
            input, grad_output, grad_weight, 
            in_features, out_features, batch_size, 
            1.0f, 0.0f);
        // grad_bias(1, o) = ones(1, b) * grad_output(b, o)
        float* d_ones;
        cudaMalloc(&d_ones, batch_size * sizeof(float));
        cudaMemset(d_ones, 0, batch_size * sizeof(float));
        fill_elements<<<(batch_size + 255)/256, 256>>>(d_ones, batch_size, 1.0f);
        gemm_gpu(TransposeType::Transpose, TransposeType::NoTranspose,
            d_ones, grad_output, grad_bias, 
            1, out_features, batch_size, 
            1.0f, 0.0f);
        cudaFree(d_ones);
    }
    

//Task2 : Convolutional Layer

__global__ void im2col_kernel(float* input_img, float* input_col,
            int batch_size, int in_channels, int height, int width){
    //Assume stride=1, padding=1, kernel_size=3
    // input_img(b, c, h, w) -> input_col()
    int col_h = height*width;
    int col_w = 3*3*in_channels;
    int col_row = blockIdx.x * blockDim.x + threadIdx.x;
    int col_col = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    if (col_row >= col_h || col_col >= col_w || batch >= batch_size) return;
    
    //计算输出位置
    int h_out = col_row / width;
    int w_out = col_row % width;
    //计算输入位置
    int channel = col_col / 9;
    int kernal_idx = col_col % 9;
    int kh = kernal_idx / 3;
    int kw = kernal_idx % 3;
    int h_in = h_out + kh - 1;
    int w_in = w_out + kw - 1;
    float value = 0.0f;
    if (h_in >=0 && h_in < height && w_in >=0 && w_in < width){
        int input_idx = batch * in_channels * height * width +
                        channel * height * width +
                        h_in * width +
                        w_in;
        value = input_img[input_idx];
    }
    int output_idx = batch * col_h * col_w +
                     col_row * col_w +
                     col_col;
    input_col[output_idx] = value;
}

void im2col(float* input_img, float* input_col,
            int batch_size, int in_channels, int height, int width,
            cudaStream_t stream){
    //Assume stride=1, padding=1, kernel_size=3
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;
    dim3 block(16, 16);
    dim3 grid((col_h + block.x -1)/block.x,
              (col_w + block.y -1)/block.y,
               batch_size);
    im2col_kernel<<<grid, block,0,stream>>>(input_img, input_col,
        batch_size, in_channels, height, width);
}

// BHWC -> BCHW
__global__ void nhwc_to_nchw_kernal(const float* bhwc, float* bchw,
                                         int batch, int out_channels, int height, int width){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * height * width;
    if (idx >= total) return;
    int oc = idx % out_channels;
    int r = idx / out_channels;
    int col_h = height * width;
    int b = r/col_h;
    int col_row = r % col_h;
    int h = col_row / width;
    int w = col_row % width;
    int out_idx = (b * out_channels * height * width) +
                  (oc * height * width) +
                  (h * width) +
                  w;
    bchw[out_idx] = bhwc[idx];
}

inline void nhwc_to_nchw(const float* bhwc, float* bchw,
                           int batch, int out_channels, int height, int width,
                           cudaStream_t stream){
    int total = batch * out_channels * height * width;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    nhwc_to_nchw_kernal<<<gridSize, blockSize, 0, stream>>>(bhwc, bchw,
                                                                 batch, out_channels, height, width);
    // 可选调试检查：
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) std::cerr << "outcol_to_nchw kernel launch failed: " << cudaGetErrorString(err) << std::endl;
}

// BCHW -> BHWC 
__global__ void nchw_to_nhwc_kernel(const float* bchw, float* bhwc,
                                      int batch, int out_channels, int height, int width){
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * out_channels * height * width;
    if (idx >= total) return;

    // 依据 NCHW 线性布局恢复坐标： idx = (((b * C + oc) * H) + h) * W + w
    int w = idx % width;
    size_t t = idx / width;
    int h = t % height;
    t = t / height;
    int oc = t % out_channels;
    int b = t / out_channels;

    int col_h = height * width;
    int outcol_idx = (b * col_h + h * width + w) * out_channels + oc;
    bhwc[outcol_idx] = bchw[idx];
}

inline void nchw_to_nhwc(const float* nchw, float* nhwc,
                           int batch, int out_channels, int height, int width,
                           cudaStream_t stream){
    if (batch <= 0 || out_channels <= 0 || height <= 0 || width <= 0) return;
    size_t total = (size_t)batch * out_channels * height * width;
    int blockSize = 256;
    size_t gridSize64 = (total + blockSize - 1) / blockSize;
    int gridSize = (int)std::min(gridSize64, (size_t)INT_MAX);
    nchw_to_nhwc_kernel<<<gridSize, blockSize, 0, stream>>>(nchw, nhwc,
                                                              batch, out_channels, height, width);
    // 可选调试：
    // cudaStreamSynchronize(stream);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) std::cerr << "nchw_to_nhwc kernel failed: " << cudaGetErrorString(err) << std::endl;
}

__global__ void col2im_kernal( float* grad_col, float* grad_input, 
           int batch_size, int in_channels, int height, int width){
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;
    int col_row = blockIdx.x * blockDim.x + threadIdx.x;
    int col_col = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    if (col_row >= col_h || col_col >= col_w || batch >= batch_size) return;

    //计算输出位置
    int h_out = col_row / width;
    int w_out = col_row % width;
    //计算输入位置
    int channel = col_col / 9;
    int kernal_idx = col_col % 9;
    int kh = kernal_idx / 3;
    int kw = kernal_idx % 3;
    int h_in = h_out + kh - 1;
    int w_in = w_out + kw - 1;
    if (h_in >=0 && h_in < height && w_in >=0 && w_in < width){
        int input_idx = batch * in_channels * height * width +
                        channel * height * width +
                        h_in * width +
                        w_in;
        int col_idx = batch * col_h * col_w +
                      col_row * col_w +
                      col_col;
        float grad_val = grad_col[col_idx];
        atomicAdd(&grad_input[input_idx], grad_val);
    }
}

void col2im(float* grad_col, float* grad_input, 
           int batch_size, int in_channels, int height, int width,cudaStream_t stream){
    size_t im_size = (size_t)batch_size * in_channels * height * width;
    cudaError_t err = cudaMemsetAsync(grad_input, 0, im_size*sizeof(float), stream);
    if (err != cudaSuccess){
        std::cerr << "col2im cudaMemsetAsync failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;
    dim3 block(16, 16);
    dim3 grid((col_h + block.x - 1)/block.x,
              (col_w + block.y - 1)/block.y,
               batch_size);
    col2im_kernal<<<grid, block,0,stream>>>(grad_col, grad_input,
        batch_size, in_channels, height, width);
}

void forward_conv2d(float* input, float* output, float* filter,
                int batch_size, int out_channels, int in_channels, int height, int width,
                cudaStream_t stream){
    //Assume stride=1, padding=1, kernel_size=3
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;
    float* d_input_col;
    cudaMalloc(&d_input_col, batch_size * col_h * col_w * sizeof(float));
    im2col(input, d_input_col, batch_size, in_channels, height, width, stream);

    float* d_output_col;
    cudaMalloc(&d_output_col, batch_size * col_h * out_channels * sizeof(float));

    //outcol(batch_size, col_h, out_channels) = input_col(batchsize, col_h, col_w) * filter(out_channels, in_channels, 3, 3) ^T
    //outcol((batch_size*col_h), out_channels) = input_col((batchsize*col_h), col_w) * filter(out_channels, (in_channels*3*3)) ^T
    gemm_gpu(TransposeType::NoTranspose, TransposeType::Transpose,
        d_input_col, filter, d_output_col,
        batch_size * col_h, out_channels, col_w,
        1.0f, 0.0f, stream);
    //move outcol(batch_size, height, weight out_channels) to output(batch_size, out_channels, height, width)
    nhwc_to_nchw(d_output_col, output, batch_size, out_channels, height, width, stream);
    cudaFree(d_output_col);
    cudaFree(d_input_col);
}

void backward_conv2d(float* input, float* filter,
        int batch_size, int out_channels, int in_channels, int height, int width,
        float* grad_input, float* grad_output, float* grad_filter,
        cudaStream_t stream){
    // Assume stride=1, padding=1, kernel_size=3
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;

    // 1) im2col(input) -> d_input_col
    float* d_input_col = nullptr;
    cudaMalloc(&d_input_col, (size_t)batch_size * col_h * col_w * sizeof(float));
    im2col(input, d_input_col, batch_size, in_channels, height, width, stream);

    // 2) convert grad_output (NCHW) -> outcol (batch*col_h, out_channels)
    float* d_grad_outcol = nullptr;
    cudaMalloc(&d_grad_outcol, (size_t)batch_size * col_h * out_channels * sizeof(float));
    nchw_to_nhwc(grad_output, d_grad_outcol, batch_size, out_channels, height, width, stream);

// // 调试片段：在调用 nchw_to_nhwc(...) 后插入（仅用于调试）
// cudaStreamSynchronize(stream); // 确保转换完成

// auto read_dev_one = [](const float* dptr, size_t idx){
//     float v=0.0f;
//     cudaMemcpy(&v, dptr + idx, sizeof(float), cudaMemcpyDeviceToHost);
//     return v;
// };

// int B = batch_size;
// int C = out_channels;
// int H = height;
// int W = width;
// int col_h_ = H * W;

// // 检查若干随机或固定位置
// std::vector<std::tuple<int,int,int,int>> checks = {
//     {0, 0, 0, 0},
//     {0, 1, 0, 1},
//     {0, C-1, H-1, W-1},
//     {B-1, C-1, H-1, W-1}
// };
// for (auto &t : checks){
//     int b,oc,h,w;
//     std::tie(b,oc,h,w) = t;
//     size_t nchw_idx   = ((size_t)b * C + oc) * H * W + (size_t)h * W + w;
//     size_t outcol_idx = ((size_t)b * col_h_ + (size_t)h * W + w) * C + oc;
//     float v_nchw = read_dev_one(grad_output, nchw_idx);
//     float v_out  = read_dev_one(d_grad_outcol, outcol_idx);
//     printf("check (b=%d,oc=%d,h=%d,w=%d) nchw[%zu]=%f outcol[%zu]=%f\n",
//            b,oc,h,w, nchw_idx, v_nchw, outcol_idx, v_out);
// }

    // 3) dW = d_grad_outcol^T * input_col
    // shapes: d_grad_outcol (batch*col_h, out_channels), d_input_col (batch*col_h, col_w)
    gemm_gpu(TransposeType::Transpose, TransposeType::NoTranspose,
        d_grad_outcol, d_input_col, grad_filter,
        out_channels, col_w, batch_size * col_h,
        1.0f, 0.0f, stream);

    // 4) dInput: grad_input_col = d_grad_outcol * filter
    float* d_grad_input_col = nullptr;
    cudaMalloc(&d_grad_input_col, (size_t)batch_size * col_h * col_w * sizeof(float));

    gemm_gpu(TransposeType::NoTranspose, TransposeType::NoTranspose,
        d_grad_outcol, filter, d_grad_input_col,
        batch_size * col_h, col_w, out_channels,
        1.0f, 0.0f, stream);
    // ensure GEMM finished before using d_grad_input_col on the same stream
    cudaStreamSynchronize(stream);

    // 5) col2im: accumulate grad_input_col -> grad_input (NCHW)
    col2im(d_grad_input_col, grad_input, batch_size, in_channels, height, width, stream);

    // 6) free temporaries
    cudaFree(d_grad_input_col);
    cudaFree(d_input_col);
    cudaFree(d_grad_outcol);
}

//Task 3: Max Pooling Layer

__global__ void max_pool_forward_kernel(const float* input, float* output, float* mask,
    int batch_size, int in_channels, int in_h, int in_w, int out_h, int out_w){
    //Assume kernel_size=2, stride=2
    int nthreads = batch_size * in_channels * out_h * out_w;
    CUDA_KERNAL_LOOP(idx, nthreads){
        int b = idx / (in_channels * out_h * out_w);  //batch index
        int c = (idx % (in_channels * out_h * out_w)) / (out_h * out_w); //channel index
        int h = (idx % (out_h * out_w)) / out_w; //height index
        int w = idx % out_w; //width index
        int h_in = h * 2;
        int w_in = w * 2;
        int max_idx = -1;
        float max_val = -INT32_MAX;
        for (int kh = 0; kh < 2; kh++){
            for (int kw = 0; kw < 2; kw++){
                int h_idx = h_in + kh;
                int w_idx = w_in + kw;
                if (h_idx >= 0 && h_idx < in_h && w_idx >= 0 && w_idx < in_w){
                    int input_idx = b * in_channels * in_h * in_w + c * in_h * in_w + h_idx * in_w + w_idx; 
                    float val = input[input_idx];
                    if (val > max_val){
                        max_val = val;
                        max_idx = input_idx;
                    }
                }
            }
        }
        output[idx] = max_val;
        mask[idx] = (float)max_idx;
    }
}

void forward_maxpool(const float* input, float* output, float* mask,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_h, int out_w, cudaStream_t stream){
        //Assume kernel_size=2, stride=2
        int nthreads = batch_size * in_channels * out_h * out_w;
        int bs = 256;
        int gs = (nthreads + bs - 1) / bs;
        max_pool_forward_kernel<<<gs, bs, 0, stream>>>(input, output, mask,
            batch_size, in_channels, in_h, in_w, out_h, out_w);
    }

__global__ void max_pool_backward_kernel(const float* grad_output, const float* mask, float* grad_input,
    int batch_size, int in_channels, int in_h, int in_w, int out_h, int out_w){
        int nthreads = batch_size * in_channels * out_h * out_w;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nthreads) return;
        int in_idx = (int)mask[idx];
        size_t in_size = (size_t)batch_size * in_channels * in_h * in_w;
        if (in_idx >= 0 && in_idx < in_size){
            atomicAdd(&grad_input[in_idx], grad_output[idx]);
        }
    }

void backward_maxpool(const float* grad_output, const float* mask, float* grad_input,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_h, int out_w, cudaStream_t stream){
        //Assume kernel_size=2, stride=2
        int nthreads = batch_size * in_channels * out_h * out_w;
        int bs = 256;
        int gs = (nthreads + bs - 1) / bs;
        //initialize grad_input to zero
        size_t im_size = (size_t)batch_size * in_channels * in_h * in_w;
        cudaError_t err = cudaMemsetAsync(grad_input, 0, im_size*sizeof(float), stream);
        //scatter grad_output to grad_input according to mask
        max_pool_backward_kernel<<<gs, bs, 0, stream>>>(grad_output, mask, grad_input,
            batch_size, in_channels, in_h, in_w, out_h, out_w);
    }

//Task4: Softmax Layer
__global__ void softmax_forward_kernel(const float* input, float* output,
    int batch_size, int num_classes){
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float buf[]; //用于reduce
    const float* input_row = input + row * num_classes;
    float* output_row = output + row * num_classes;

    float local_max = -INT32_MAX;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x){
        if (input_row[i] > local_max) local_max = input_row[i];
    }
    buf[threadIdx.x] = local_max;
    __syncthreads();
    //reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s){
            if (buf[threadIdx.x + s] > buf[threadIdx.x]){
                buf[threadIdx.x] = buf[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float row_max = buf[0];
    //compute exp 
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x){
        float val = expf(input_row[i] - row_max);
        output_row[i] = val;
        local_sum += val;
    }
    buf[threadIdx.x] = local_sum;
    __syncthreads();
    //reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s){
            buf[threadIdx.x] += buf[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = buf[0];
    //compute softmax
    float inv_row_sum = (row_sum != 0.0f) ? (1.0f / row_sum) : 0.0f;;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x){
        output_row[i] *= inv_row_sum;
    }
}

void forward_softmax(const float* input, float* output,
    int batch_size ,int num_classes, cudaStream_t stream){
    //input shape (batch_size, num_classes)
    //output shape (batch_size, num_classes)
    int bs = 256;
    dim3 grid(batch_size);
    softmax_forward_kernel<<<grid, bs, /*shared memory*/bs * sizeof(float), stream>>>(input, output,
        batch_size, num_classes);
}

//Task5: Cross Entropy Loss Layer

void backward_cross_entropy(const float* softmax_output, const int* labels,
    int batch_size, int num_classes, float* grad_input, cudaStream_t stream){
        //softmax_output shape (batch_size, num_classes)
        //labels shape (batch_size)
        //grad_input shape (batch_size, num_classes)
        //grad_input = softmax_output
        cudaMemcpyAsync(grad_input, softmax_output, 
            batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        //grad_input(b, c) -= 1 if c == labels[b]
        int bs = 256;
        int gs = (batch_size + bs - 1) / bs;
        subtract_labels<<<gs, bs, 0, stream>>>(grad_input, labels, batch_size, num_classes);
        //grad_input /= batch_size
        scale_tensor<<<gs, bs, 0, stream>>>(grad_input, batch_size * num_classes, 1.0f / batch_size);
    }
#endif //LAYERS_H