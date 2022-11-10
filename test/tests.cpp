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
    tests.execMLPRegressorTests();
}
