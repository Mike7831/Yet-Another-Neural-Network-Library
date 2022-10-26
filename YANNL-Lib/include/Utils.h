#ifndef UTILS_H
#define UTILS_H

#include <iostream>     // std::ostream
#include <vector>       // std::vector
#include <algorithm>    // std_max_element & std::min_element
#include <random>       // std::random_device
#include <windows.h>    // GetConsoleCursorInfo & SetConsoleCursorInfo

//! Prints a vector to the provided output stream.
//! @param os Output stream to print to.
//! @param vect Vector to print.t
//! @returns Output stream to print to.
template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vect)
{
    os << "[";
    bool first_iter = true;

    for (auto val : vect)
    {
        if (first_iter)
        {
            first_iter = false;
        }
        else
        {
            os << " | ";
        }

        os << val;
    }

    os << "]";

    return os;
}

//! Performs a min-max normalization of a vector.
//! @param vect The vector to normalize.
//! @returns Normalized vector.
template <class T>
std::vector<double> normalizeVect(const std::vector<T>& vect)
{
    T max = *std::max_element(vect.begin(), vect.end());
    T min = *std::min_element(vect.begin(), vect.end());
    double diff = max - min;

    std::vector<double> normVect;

    std::for_each(vect.cbegin(), vect.cend(),
        [&](const auto& item)
        {
            normVect.push_back((item - min) / diff);
        });

    return normVect;
}

//! Converts a one-byte label (uint8_t) as a classification vector of size
//! minLabel to maxLabel. For example label 1 for minLabel = 0 and maxLabel = 3
//! is converted to { 0, 1, 0, 0 } (4 items from 0 to 3). If the label provided
//! is not within the range then the vector will be all set to 0s.
//! @param label Label for which value is 1. Other labels have a 0-value.
//! @param minLabel Lower bound of the vector.
//! @param maxLabel Upper bound of the vector.
//! @returns Vector as illustrated in the example above.
std::vector<double> convertLabelToVect(uint8_t label, size_t minLabel, size_t maxLabel)
{
    std::vector<double> output((maxLabel > minLabel ?
        (maxLabel - minLabel) :
        (minLabel - maxLabel)) + 1);

    for (uint8_t n = 0; n <= maxLabel; n++)
    {
        if (n == label)
        {
            output[n] = 1.0;
        }
        else
        {
            output[n] = 0.0;
        }
    }

    return output;
}

//! Show or hide the console cursor to avoid it blinking when updating the
//! console output fast.
//! @param showFlag True to show the cursor, false to hide.
void ShowConsoleCursor(bool showFlag)
{
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);

    CONSOLE_CURSOR_INFO cursorInfo;

    GetConsoleCursorInfo(out, &cursorInfo);
    cursorInfo.bVisible = showFlag; // set the cursor visibility
    SetConsoleCursorInfo(out, &cursorInfo);
}

//! @brief Class for generating seeds. It uses itself a seed to make sure
//! the seed sequence is always the same if the user wants it. If the
//! user does not want to generate always the same sequence he/she can provide a custom seed.
class SeedGenerator
{
public:
    //! @param useCustomSeed Tells whether to use a custom seed
    //! @param seed Custom seed, used only if useCustomSeed is true
    SeedGenerator(bool useCustomSeed = false, unsigned int seed = 0)
    {
        if (!useCustomSeed)
        {
            std::random_device rng;
            seed = rng();
        }

        mtGenerator.seed(seed);
    }

    unsigned int seed()
    {
        return mtGenerator();
    }

    friend std::ostream& operator<<(std::ostream& os, const SeedGenerator& generator);

    friend std::istream& operator>>(std::istream& is, SeedGenerator& generator)
    {
        is >> generator.mtGenerator;
        return is;
    }

private:
    std::mt19937 mtGenerator;
};

std::ostream& operator<<(std::ostream& os, const SeedGenerator& generator)
{
    os << generator.mtGenerator;
    return os;
}

#endif // UTILS_H
