#include "tensor.h"
#include "layers.h"

void testFC(){
    std::cout << "------Task1: fully-connected layer------" << std::endl;
    std::cout << "Testing forward_fc function..." << std::endl <<std::endl;
    Tensor<float> input({5,3}, Device::GPU); //batch_size=5, in_features=3
    Tensor<float> weight({3,2}, Device::GPU); //batch_size=5, out_features=2
    Tensor<float> output({5,2}, Device::GPU);
    matrix_init(input.get_data(), 5, 3);
    matrix_init(weight.get_data(), 3, 2);
    fill_elements<<<1,16>>>(output.get_data(),10,0);
    std::cout << "input:" <<  std::endl;
    input.print();
    std::cout << "weight:" <<  std::endl;
    weight.print();
    Tensor<float> bias({1,2}, Device::GPU);
    // fill_elements<<<2,2>>>(bias.get_data(),2,0);
    matrix_init(bias.get_data(), 1, 2);
    std::cout << "bias:" <<  std::endl;
    bias.print();

    forward_fc(input.get_data(),output.get_data(), weight.get_data(), bias.get_data(),
                5, 2, 3);
    std::cout << "output:" <<  std::endl;
    output.print();
    std::cout << std::endl;

    std::cout << "Testing backward_fc function..." << std::endl << std::endl;
    Tensor<float> grad_output({5,2}, Device::GPU);
    matrix_init(grad_output.get_data(), 5, 2);
    Tensor<float> grad_input({5,3}, Device::GPU);
    Tensor<float> grad_weight({3,2}, Device::GPU);
    Tensor<float> grad_bias({1,2}, Device::GPU);
    fill_elements<<<1,16>>>(grad_input.get_data(),15,0);
    fill_elements<<<1,16>>>(grad_weight.get_data(),6,0);
    fill_elements<<<1,16>>>(grad_bias.get_data(),2,0);
    std::cout << "grad_output:" <<  std::endl;
    grad_output.print();
    backward_fc(input.get_data(), weight.get_data(), bias.get_data(),
                5, 2, 3,
                grad_input.get_data(), grad_output.get_data(), grad_weight.get_data(), grad_bias.get_data());
    std::cout << "grad_input:" << std::endl;
    grad_input.print();
    std::cout << "grad_weight:" << std::endl;
    grad_weight.print(); 
    std::cout << "grad_bias:" << std::endl;
    grad_bias.print();
}

void testConv2d(){
    std::cout << "------Task2: convolution layer------" << std::endl;
    std::cout << "Testing forward_conv2d function..." << std::endl <<std::endl;    
    Tensor<float> input({1,3,4,4},Device::GPU); //batch_size, inchannel, height, width
    Tensor<float> weight({2,3,3,3},Device::GPU); // outchannel, inchannel, kernel_size, kernel_size
    Tensor<float> output({1,2,4,4},Device::GPU); // batch_size, out_channel, height, width
    matrix_init(input.get_data(), input.get_size(),1);
    matrix_init(weight.get_data(), weight.get_size(),1);
    forward_conv2d(input.get_data(), output.get_data(), weight.get_data(),
                /*batch_size*/1, /*out_channel*/2, /*in_channel*/3, /*height*/4, /*width*/4, /*stream*/0);
    std::cout << "input:" <<  std::endl;
    input.print();
    std::cout << "weight:" <<  std::endl;
    weight.print();
    std::cout << "output:" <<  std::endl;
    output.print();
    std::cout << std::endl;

    std::cout << "Testing backward_conv2d function..." << std::endl << std::endl;
    Tensor<float> grad_output({1,2,4,4},Device::GPU);
    matrix_init(grad_output.get_data(), grad_output.get_size(),1);
    Tensor<float> grad_input({1,3,4,4},Device::GPU);
    Tensor<float> grad_weight({2,3,3,3},Device::GPU);
    fill_elements<<<1,16>>>(grad_input.get_data(),grad_input.get_size(),0);
    fill_elements<<<1,16>>>(grad_weight.get_data(),grad_weight.get_size(),0);
    backward_conv2d(input.get_data(), weight.get_data(), 
        /*batch_size*/1, /*out_channel*/2, /*in_channel*/3, /*height*/4, /*width*/4,
        grad_input.get_data(), grad_output.get_data(), grad_weight.get_data(),
        /*stream*/0
    );
    std::cout << "grad_output:" <<  std::endl;
    grad_output.print();
    std::cout << "grad_input:" << std::endl;
    grad_input.print();
    std::cout << "grad_weight:" << std::endl;
    grad_weight.print();
}

int main(){

    // testFC();
    testConv2d();
}