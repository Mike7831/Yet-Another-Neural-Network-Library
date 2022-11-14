//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#include "UnitTests.h"

using namespace YANNL;

int main(int argc, char* argv[])
{
    YANNL_UnitTests tests;
    tests.execExceptionTests();
    tests.execNeuralNetworkTests();
    tests.execOtherTests();
    tests.execMnistTests();
    tests.execXMLTests();
    tests.execMLPTests();
}
