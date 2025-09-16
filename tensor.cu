#include <iostream>
#include <vector>
#include <cuda.h>
#include <memory>

enum class Device{
    CPU,
    GPU
};

class Tensor{
private:
    std::vector<int> shape;
    std::vector<int> strides;
    std::unique_ptr<float[]> h_data;
    float* d_data = nullptr;
    Device device;
    int size{1};

public:
    Tensor(std::vector<int> s, device d):shape(s), device(d) {
        for (int dim: shape){
            size *= dim;
        }
        int k = size;
        for (int dim: shape){
            k = k/dim;
            strides.push_back(k);
        }
        if (device == Device::CPU){
            h_data = std::make_unique<float[]>(size);
        }
        else{
            cudaMalloc(&d_data, size * sizeof(float));
        }
    }

    float* get_data(){
        if (device == Device::CPU){
            return h_data.get();
        }
        else{
            return d_data;
        }
    }

    int get_size(){
        return size;
    }

    ~Tensor(){
        if (device == Device::GPU){
            cudaFree(d_data);
        }
        else if (device == Device::CPU){
            h_data.reset();
        }
    }

    Tensor cpu(){
        if (device == Device::CPU){
            return *this;
        }
        Tensor t(shape, Device::CPU);
        cudaMemcpy(t.get_data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        return t;
    }

    Tensor gpu(){
        if (device == Device::GPU){
            return *this;
        }
        Tensor t(shape, Device::GPU);
        cudaMemcpy(t.get_data(), h_data.get(), size * sizeof(float), cudaMemcpyHostToDevice);
        return t;
    }

    //辅助函数
    void print(){
        if (device == Device::CPU){
            for (int i = 0; i < size; i++){
                std::cout << h_data[i] << " ";
                for (int dim: strides){
                    if (dim!= 1 && (i+1) % dim == 0){
                        std::cout << std::endl;
                    }
                }
            }
        }
        else if (device == Device::GPU){
            Tensor t = this->cpu();
            t.print();
        }
    }

};

int main(){
    Tensor t({3,2}, Device::CPU);
    float* data = t.get_data();
    for (int i = 0; i < t.get_size(); ++i){
        data[i] = i;
    }
    t.print();
}