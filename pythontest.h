#include <iostream>

#ifdef COLOR_STDOUT
#define BOLD   "\033[1m"
#define RED    "\033[31m"
#define GREEN  "\033[32m"
#define YELLOW "\033[33m"
#define BLUE   "\033[34m"
#define RESET  "\033[0m"
#else
#define BOLD   ""
#define RED    ""
#define GREEN  ""
#define YELLOW ""
#define BLUE   ""
#define RESET  ""
#endif

#ifdef PYTEST_ON
#define PYTEST  ""
#else
#define PYTEST  return
#endif

std::string run_python(const std::string& cmd) {
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "ERROR";
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}


void runPy(const char * root){
    PYTEST;
    std::cout << BOLD << GREEN << "**Pytorch Result**" << RESET << std::endl;
    std::cout << GREEN << run_python(root) << RESET;
    std::cout << BOLD << GREEN << "**CORRECT**" << RESET << std::endl;
    std::cout << std::endl;
}
