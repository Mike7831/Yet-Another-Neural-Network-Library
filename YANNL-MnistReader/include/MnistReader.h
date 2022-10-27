#ifndef YANNL_MNIST_READER_H
#define YANNL_MNIST_READER_H

#include "Utils.h"
#include <fstream>  // std::ifstream
#include <sstream>  // std::ostringstream
#include <vector>   // std::vector

class MnistReader
{
public:
    typedef std::vector<std::vector<uint8_t>> ImageContainer;
    typedef std::vector<std::vector<double>> NormalizedImageContainer;
    typedef std::vector<uint8_t> LabelContainer;

    struct MnistFileAttrs
    {
        int32_t count = 0; // Number of images
        int32_t rowsN = 0; // Number of rows
        int32_t colsN = 0; // Number of columns
    };

    static MnistFileAttrs readMnist(const std::string& filename, ImageContainer& images)
    {
        images.clear();
        std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);

        if (!file)
        {
            throw std::ios_base::failure(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Read MNIST] Error opening file " << filename << ".").str()
            );
        }

        auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        int32_t magic = 0;
        MnistFileAttrs attrs;

        file.read((char*)&magic, sizeof(magic));
        magic = swapEndian(magic);

        file.read((char*)&attrs.count, sizeof(attrs.count));
        attrs.count = swapEndian(attrs.count);

        if (magic == 0x803) // Images
        {
            file.read((char*)&attrs.rowsN, sizeof(attrs.rowsN));
            attrs.rowsN = swapEndian(attrs.rowsN);

            file.read((char*)&attrs.colsN, sizeof(attrs.colsN));
            attrs.colsN = swapEndian(attrs.colsN);

            // File size should be at least one byte/char per pixel + 4 integer headers
            // Structure: http://yann.lecun.com/exdb/mnist
            if (size < static_cast<long long>(attrs.count) * attrs.rowsN * attrs.colsN + 16)
            {
                throw std::ios_base::failure(static_cast<const std::ostringstream&>(std::ostringstream()
                    << "[Read MNIST] " << filename << " seems corrupted; not large enough.").str()
                );
            }

            for (int32_t n = 0; n < attrs.count; n++)
            {
                std::vector<uint8_t> image;

                for (int32_t r = 0; r < attrs.rowsN; r++)
                {
                    for (int32_t c = 0; c < attrs.colsN; c++)
                    {
                        uint8_t pixel = 0;
                        file.read((char*)&pixel, sizeof(pixel));
                        image.push_back(pixel);
                    }
                }

                images.push_back(image);
            }
        }
        else
        {
            attrs.count = 0;
        }

        return attrs;
    }

    //! @return Number of labels read
    static int32_t readMnist(const std::string& filename, LabelContainer& labels)
    {
        labels.clear();
        std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);

        if (!file)
        {
            throw std::ios_base::failure(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Read MNIST] Error opening file " << filename << ".").str()
            );
        }

        auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        int32_t magic = 0;
        MnistFileAttrs attrs;

        file.read((char*)&magic, sizeof(magic));
        magic = swapEndian(magic);

        file.read((char*)&attrs.count, sizeof(attrs.count));
        attrs.count = swapEndian(attrs.count);

        if (magic == 0x801) // Labels
        {
            // File size should be at least one byte/char per label + 2 integer headers
            // Structure: http://yann.lecun.com/exdb/mnist
            if (size < static_cast<long long>(attrs.count) + 8)
            {
                throw std::ios_base::failure(static_cast<const std::ostringstream&>(std::ostringstream()
                    << "[Read MNIST] " << filename << " seems corrupted; not large enough.").str()
                );
            }

            for (int32_t n = 0; n < attrs.count; n++)
            {
                uint8_t label = 0;
                file.read((char*)&label, sizeof(label));
                labels.push_back(label);
            }
        }
        else
        {
            attrs.count = 0;
        }

        return attrs.count;
    }

    static void displayMnist(const ImageContainer& images, std::ostream& os, size_t beginN, size_t endN, MnistFileAttrs attrs)
    {
        os << "Dataset contains " << attrs.count << " images of "
            << attrs.rowsN << "x" << attrs.colsN << "\n"
            << "Displaying images from " << beginN << " to " << endN << "\n";

        for (size_t n = beginN; n < endN && n < static_cast<size_t>(attrs.count); n++)
        {
            os << "--- [Image " << n << "] --- \n";

            for (int32_t r = 0; r < attrs.rowsN; r++)
            {
                for (int32_t c = 0; c < attrs.colsN; c++)
                {
                    int32_t coord = c + r * attrs.colsN;

                    if (images[n][coord] == 0)
                    {
                        os << " ";
                    }
                    else
                    {
                        os << "x";
                    }
                }

                os << "\n";
            }

            os << "------------------ \n";
        }
    }

    static void displayMnist(const LabelContainer& labels, std::ostream& os, size_t beginN, size_t endN, int32_t count)
    {
        os << "Dataset contains " << count << " labels \n"
            << "Displaying labels from " << beginN << " to " << endN << "\n";

        for (size_t n = beginN; n < endN && n < static_cast<size_t>(count); n++)
        {
            os << "[Label " << n << "] " << (int)labels[n] << "\n";
        }
    }

    static NormalizedImageContainer normalize(const ImageContainer& images)
    {
        NormalizedImageContainer normImages;

        std::for_each(images.cbegin(), images.cend(),
            [&](const auto& image)
            {
                normImages.push_back(normalizeVect(image));
            });

        return normImages;
    }

private:
    static uint32_t swapEndian(uint32_t n)
    {
        uint8_t ch1 = n & 0xFF;
        uint8_t ch2 = (n >> 8) & 0xFF;
        uint8_t ch3 = (n >> 16) & 0xFF;
        uint8_t ch4 = (n >> 24) & 0xFF;

        return (static_cast<uint32_t>(ch1) << 24)
            + (static_cast<uint32_t>(ch2) << 16)
            + (static_cast<uint32_t>(ch3) << 8)
            + (static_cast<uint32_t>(ch4));
    }

    MnistReader() {};
};

#endif // YANNL_MNIST_READER_H