#include "demo.h"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: ./app lib_path graph_path parameters_path\n";
        return -1;
    }

    std::string lib_path(argv[1]);
    std::string graph_path(argv[2]);
    std::string param_path(argv[3]);

    return Apply(lib_path, graph_path, param_path);
}
