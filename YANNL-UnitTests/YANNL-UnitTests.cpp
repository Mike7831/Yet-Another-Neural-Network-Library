#include "UnitTests.h"

//#include "MLP.h"

#include <iomanip> // std::setprecision

using namespace YANNL;

int main(int argc, char* argv[])
{
    YANNL_UnitTests tests;
    tests.execExceptionTests();
    tests.execNeuralNetworkTests();
    tests.execOtherTests();
}
