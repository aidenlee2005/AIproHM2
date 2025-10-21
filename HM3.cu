#include "testcase.h"
#include <fstream>
#include <cstring>

int flag = 0;

/*
编译：
nvcc HM3.cu -o [name] -lcublas -lcurand
选项：
（1）在其后加 -DCOLOR_STDOUT 选项，可以将输出颜色化显示。
（2）在其后加 -DPYTEST_ON 选项，可以输出Pytorch的正确答案参考。

运行：
./[name] (f)
选项：
（1）省略“f”字符，则在控制台输出；
（2）加上“f“字符，则输出到"testResult.txt"文件中。

*/

int main(int argc, char* argv[]){
    std::streambuf* coutbuf = nullptr;
    std::ofstream out;
    if (argc == 2 && std::strcmp(argv[1], "f") == 0){
        flag = 1;
        out.open("testResult.txt");
        coutbuf = std::cout.rdbuf(); //save old buf
        std::cout.rdbuf(out.rdbuf());
    }

    std::cout << BOLD << RED << "Running HM3 test cases..." << RESET << std::endl;
    testFC();
    testConv2d();
    testMaxpool2d();
    testSoftmaxAndCrossEntropy();
    std::cout << BOLD << GREEN << "All test cases passed!" << RESET << std::endl;

    std::cout.flush();

    if (flag == 1){
        std::cout.rdbuf(coutbuf);
    }
}