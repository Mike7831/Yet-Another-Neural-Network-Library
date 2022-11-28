//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#include "UnitTests.h"
#include <chrono>   // std::chrono

using namespace YANNL;

int main(int argc, char* argv[])
{
    auto t0 = std::chrono::high_resolution_clock::now();

    YANNL_UnitTests tests;
    //tests.execExceptionTests();
    //tests.execNeuralNetworkTests();
    //tests.execOtherTests();
    //tests.execMnistTests();
    //tests.execXMLTests();
    //tests.execMLPTests();
    tests.execBatchTrainingTests();

    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "Testing completed in " + std::to_string(elapsed) + " ms.";
}
