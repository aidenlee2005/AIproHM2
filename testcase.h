#include "tensor.h"
#include "layers.h"
#include "pythontest.h"


void testFC(){
    std::cout << BOLD << BLUE << "---------Task1: fully-connected layer---------" << RESET << std::endl;
    std::cout << BOLD << YELLOW <<  "Testing forward_fc function..." << RESET << std::endl <<std::endl;

    Tensor<float> input({5,3}, Device::GPU); //batch_size=5, in_features=3
    Tensor<float> weight({3,2}, Device::GPU); //batch_size=5, out_features=2
    Tensor<float> output({5,2}, Device::GPU);
    Tensor<float> bias({1,2}, Device::GPU);

    matrix_init_float(input.get_data(), 15);
    matrix_init_float(weight.get_data(), 6);
    fill_elements<<<1,16>>>(output.get_data(),10,0);
    matrix_init_float(bias.get_data(), 2);

    std::cout << BOLD << "input:" << RESET << std::endl;
    input.print();
    std::cout << BOLD << "weight:" << RESET <<  std::endl;
    weight.print();
    std::cout << BOLD << "bias:"<< RESET <<  std::endl;
    bias.print();

    forward_fc(input.get_data(),output.get_data(), weight.get_data(), bias.get_data(),
                5, 2, 3);

    std::cout << BOLD << "output:" << RESET <<  std::endl;
    output.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test1_forward.py");
    std::cout << std::endl;

    std::cout << BOLD << YELLOW << "Testing backward_fc function..." << RESET << std::endl << std::endl;

    Tensor<float> grad_output({5,2}, Device::GPU);
    matrix_init_float(grad_output.get_data(), 10);
    Tensor<float> grad_input({5,3}, Device::GPU);
    Tensor<float> grad_weight({3,2}, Device::GPU);
    Tensor<float> grad_bias({1,2}, Device::GPU);
    fill_elements<<<1,16>>>(grad_input.get_data(),15,0);
    fill_elements<<<1,16>>>(grad_weight.get_data(),6,0);
    fill_elements<<<1,16>>>(grad_bias.get_data(),2,0);

    std::cout << BOLD << "grad_output:" << RESET <<  std::endl;
    grad_output.print();

    backward_fc(input.get_data(), weight.get_data(), bias.get_data(),
                5, 2, 3,
                grad_input.get_data(), grad_output.get_data(), grad_weight.get_data(), grad_bias.get_data());

    std::cout << BOLD << "grad_input:" << RESET << std::endl;
    grad_input.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test1_gradinput.py");

    std::cout << BOLD << "grad_weight:" << RESET << std::endl;
    grad_weight.print(); 
    runPy("python3 /root/autodl-tmp/HM3PyTest/test1_gradweight.py");

    std::cout << BOLD << "grad_bias:" << RESET << std::endl;
    grad_bias.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test1_gradbias.py");

    std::cout << std::endl;
}

void testConv2d(){
    std::cout << BOLD << BLUE << "---------Task2: convolution layer---------" << RESET << std::endl;
    std::cout << BOLD << YELLOW << "Testing forward_conv2d function..." << RESET << std::endl <<std::endl; 

    Tensor<float> input({1,3,4,4},Device::GPU); //batch_size, inchannel, height, width
    Tensor<float> weight({2,3,3,3},Device::GPU); // outchannel, inchannel, kernel_size, kernel_size
    Tensor<float> output({1,2,4,4},Device::GPU); // batch_size, out_channel, height, width
    matrix_init_float(input.get_data(), input.get_size());
    matrix_init_float(weight.get_data(), weight.get_size());

    forward_conv2d(input.get_data(), output.get_data(), weight.get_data(),
                /*batch_size*/1, /*out_channel*/2, /*in_channel*/3, /*height*/4, /*width*/4, /*stream*/0);

    std::cout << BOLD << "input:" << RESET <<  std::endl;
    input.print();

    std::cout << BOLD << "weight:" << RESET <<  std::endl;
    weight.print();

    std::cout << BOLD << "output:" << RESET <<  std::endl;
    output.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test2_forwardoutput.py");

    std::cout << std::endl;

    std::cout << BOLD << YELLOW << "Testing backward_conv2d function..." << RESET << std::endl << std::endl;

    Tensor<float> grad_output({1,2,4,4},Device::GPU);
    matrix_init_float(grad_output.get_data(), grad_output.get_size());
    Tensor<float> grad_input({1,3,4,4},Device::GPU);
    Tensor<float> grad_weight({2,3,3,3},Device::GPU);
    fill_elements<<<1,16>>>(grad_input.get_data(),grad_input.get_size(),0);
    fill_elements<<<1,16>>>(grad_weight.get_data(),grad_weight.get_size(),0);

    backward_conv2d(input.get_data(), weight.get_data(), 
        /*batch_size*/1, /*out_channel*/2, /*in_channel*/3, /*height*/4, /*width*/4,
        grad_input.get_data(), grad_output.get_data(), grad_weight.get_data(),
        /*stream*/0
    );

    std::cout << BOLD << "grad_output:" << RESET <<  std::endl;
    grad_output.print();

    std::cout << BOLD << "grad_input:" << RESET << std::endl;
    grad_input.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test2_gradinput.py");

    std::cout << BOLD << "grad_weight:" << RESET << std::endl;
    grad_weight.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test2_gradweight.py");

    std::cout << std::endl;
}

void testMaxpool2d(){
    std::cout << BOLD << BLUE << "---------Task3: Maxpooling layer---------" << RESET << std::endl;
    std::cout << BOLD << YELLOW << "Testing forward_maxpool2d function..." << RESET <<std::endl <<std::endl;
    
    Tensor<float> input({1,3,4,4},Device::GPU); //batch_size, inchannel, height, width
    Tensor<float> output({1,3,2,2},Device::GPU); // batch_size, out_channel, height, width
    Tensor<float> mask({1,3,2,2},Device::GPU);
    matrix_init_float(input.get_data(), input.get_size());
    
    forward_maxpool(input.get_data(), output.get_data(), mask.get_data(),
                /*batch_size*/1, /*in_channel*/3, /*in_h*/4, /*in_w*/4,
                 /*out_h*/2, /*out_w*/2, /*stream*/0);

    std::cout << BOLD << "input:" << RESET <<  std::endl;
    input.print();

    std::cout << BOLD << "output:" << RESET << std::endl;
    output.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test3_forwardoutput.py");

    std::cout << BOLD << "mask:" << RESET << std::endl;
    mask.print();
    std::cout << std::endl;

    std::cout << BOLD << YELLOW << "Testing backward_maxpool2d function..." << RESET << std::endl <<std::endl;
    
    Tensor<float> grad_output({1,3,2,2},Device::GPU);
    matrix_init_float(grad_output.get_data(), grad_output.get_size());
    Tensor<float> grad_input({1,3,4,4},Device::GPU);
    fill_elements<<<1,16>>>(grad_input.get_data(),grad_input.get_size(),0);
    
    backward_maxpool(grad_output.get_data(), mask.get_data(), grad_input.get_data(),
                /*batch_size*/1, /*in_channel*/3, /*in_h*/4, /*in_w*/4,
                 /*out_h*/2, /*out_w*/2, /*stream*/0);

    std::cout << BOLD << "grad_output:" << RESET <<  std::endl;
    grad_output.print();

    std::cout << BOLD << "grad_input:" << RESET << std::endl;
    grad_input.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test3_backward.py");

    std::cout << std::endl;
}

void testSoftmaxAndCrossEntropy(){
    std::cout << BOLD << BLUE << "---------Task4&5: Softmax and Cross-entropy loss---------" << RESET << std::endl;
    std::cout << BOLD << YELLOW << "Testing forward_softmax function..." << RESET << std::endl <<std::endl;
    
    Tensor<float> input({3,4},Device::GPU); //batch_size, classes_num
    Tensor<float> output({3,4},Device::GPU); 
    Tensor<float> grad_output({3,4},Device::GPU);
    Tensor<int> target({3},Device::GPU); // batch_size
    matrix_init_float(input.get_data(), input.get_size());
    matrix_init_int(target.get_data(), target.get_size(), 0, 4); // set target to 0,1,2,3

    std::cout << BOLD << "input:" << RESET  <<  std::endl;
    input.print();

    std::cout << BOLD  << "target:" << RESET<<  std::endl;
    target.print();

    forward_softmax(input.get_data(), output.get_data(),3,4,0);

    std::cout << BOLD  << "output:"<< RESET <<  std::endl;
    output.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test4_softmax.py");

    std::cout << std::endl;

    std::cout << BOLD << YELLOW << "Testing backward_cross_entropy function..." << RESET << std::endl <<std::endl;

    float loss;
    forward_cross_entropy(output.get_data(), target.get_data(), &loss,  3, 4, 0);

    std::cout << BOLD  << "loss:"<< RESET << std::endl << loss << std::endl;
    runPy("python3 /root/autodl-tmp/HM3PyTest/test5_loss.py");

    backward_cross_entropy(output.get_data(), target.get_data(), 3, 4, grad_output.get_data(), 0);
    
    std::cout << BOLD  << "grad_output:"<< RESET <<  std::endl;
    grad_output.print();
    runPy("python3 /root/autodl-tmp/HM3PyTest/test5_gradoutput.py");
}