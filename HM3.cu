#include "testcase.h"

int main(){
    std::cout << BOLD << RED << "Running HM3 test cases..." << RESET << std::endl;
    testFC();
    testConv2d();
    testMaxpool2d();
    testSoftmaxAndCrossEntropy();
    std::cout << BOLD << GREEN << "All test cases passed!" << RESET << std::endl;
}