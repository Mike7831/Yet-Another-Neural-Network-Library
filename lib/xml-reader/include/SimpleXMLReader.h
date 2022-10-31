#ifndef YANNL_SIMPLE_XML_READER_H
#define YANNL_SIMPLE_XML_READER_H

#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <memory>   // std::unique_ptr & std::shared_ptr
#include <algorithm> // std::for_each

namespace YANNL
{
std::string trim(std::string str)
{
    const std::string whitespaces = " \t\n\r\f\v";

    str.erase(str.find_last_not_of(whitespaces) + 1);
    str.erase(0, str.find_first_not_of(whitespaces));

    return str;
}

enum class XMLState
{
    InFile = 0,
    InOpeningTag,
    InClosingTag,
    InOpeningTagAttributes,
    InOpeningTagAttribute,
    InOpeningTagAttributeValue,
    InClosingTagAttributes,
    InValue,
    InComment
};

struct XMLNode
{
    std::string name;
    std::string value;
    std::map<std::string, std::string> attributes;
    XMLNode* parent = nullptr;
    std::vector<std::unique_ptr<XMLNode>> children;

    void inspect(std::ostream& os, size_t indent = 0) const
    {
        os << std::string(indent, ' ') << "<" << name;

        std::for_each(attributes.cbegin(), attributes.cend(),
            [&](const std::pair<std::string, std::string>& attr)
            {
                os << " " << attr.first << "=\"" << attr.second << "\"";
            });

        os << ">";

        if (children.empty())
        {
            os << value << "</" << name << ">\n";
        }
        else
        {
            os << "\n";

            if (!value.empty())
            {
                os << std::string(indent + 2, ' ') << value << "\n";
            }

            std::for_each(children.cbegin(), children.cend(),
                [&](const std::unique_ptr<XMLNode>& child)
                {
                    child->inspect(os, indent + 2);
                });

            os << std::string(indent, ' ') << "</" << name << ">\n";
        }
    }

    XMLNode* getNode(std::string path)
    {
        const char delimiter = '/';
        XMLNode* curNode = this;
        XMLNode* node = nullptr;

        // Remove first delimiter if exists
        if (!path.empty() && path[0] == delimiter)
        {
            path.erase(0, 1);
        }

        // Add a delimiter to the end if it does not exist
        if (!path.empty() && path[path.size() - 1] != delimiter)
        {
            path += delimiter;
        }

        size_t pos = 0;
        std::string token;

        if ((pos = path.find(delimiter)) != std::string::npos
            && path.substr(0, pos) == curNode->name)
        {
            // Check whether it is at the right place, i.e. the top node provided is
            // the current node. Once done, remove it from the hierarchy.
            path.erase(0, pos + 1);

            while ((pos = path.find(delimiter)) != std::string::npos)
            {
                token = path.substr(0, pos);
                size_t index = 0;

                size_t indexB = token.find_first_of('[');
                size_t indexE = token.find_first_of(']');

                if (indexB != std::string::npos && indexE != std::string::npos)
                {
                    std::stringstream sstream(token.substr(indexB + 1, indexE - indexB - 1));
                    sstream >> index;
                    token = token.substr(0, indexB);
                }

                size_t n = 0;

                for (size_t i = 0; i < curNode->children.size() && n <= index; i++)
                {
                    if (curNode->children[i]->name == token)
                    {
                        if (n == index)
                        {
                            curNode = curNode->children[i].get();

                            // If there is only one delimiter left (at the end)
                            // it means the element the user seeks is found. 
                            // Returns that element.
                            if (std::count(path.begin(), path.end(), delimiter) == 1)
                            {
                                node = curNode;
                            }
                        }

                        n++;
                    }
                }

                path.erase(0, pos + 1);
            }
        }

        return node;
    }

    std::vector<XMLNode*> getCollection(std::string path)
    {
        const char delimiter = '/';
        XMLNode* curNode = this;
        std::vector<XMLNode*> collection;

        // Remove first delimiter if exists
        if (!path.empty() && path[0] == delimiter)
        {
            path.erase(0, 1);
        }

        // Add a delimiter to the end if it does not exist
        if (!path.empty() && path[path.size() - 1] != delimiter)
        {
            path += delimiter;
        }

        size_t pos = 0;
        std::string token;

        if ((pos = path.find(delimiter)) != std::string::npos
            && path.substr(0, pos) == curNode->name)
        {
            path.erase(0, pos + 1);

            while ((pos = path.find(delimiter)) != std::string::npos)
            {
                token = path.substr(0, pos);
                size_t index = 0;

                size_t indexB = token.find_first_of('[');
                size_t indexE = token.find_first_of(']');
                bool indexFound = (indexB != std::string::npos && indexE != std::string::npos);

                if (indexFound)
                {
                    std::stringstream sstream(token.substr(indexB + 1, indexE - indexB - 1));
                    sstream >> index;
                    token = token.substr(0, indexB);
                }

                size_t n = 0;
                bool lastElement = (std::count(path.begin(), path.end(), delimiter) == 1);

                for (size_t i = 0; i < curNode->children.size() && n <= index; i++)
                {
                    if (curNode->children[i]->name == token)
                    {
                        if (!lastElement)
                        {
                            if (n == index)
                            {
                                curNode = curNode->children[i].get();
                            }

                            n++;
                        }
                        else // if (lastElement)
                        {
                            if (indexFound)
                            {
                                // e.g. /network/layers/layer[1]/neurons[0]
                                // Index provided: add only the requested node
                                if (n == index)
                                {
                                    collection.push_back(curNode->children[i].get());
                                }

                                n++;
                            }
                            else // if (!indexFound)
                            {
                                // e.g. /network/layers/layer[1]/neurons
                                // No index provided: add all the nodes
                                collection.push_back(curNode->children[i].get());
                            }
                        }
                    }
                }

                path.erase(0, pos + 1);
            }
        }

        return collection;
    }
};

std::unique_ptr<XMLNode> readXMLStream(std::istream& ifs)
{
    std::string line, tag, attr, value;
    std::map<std::string, std::string> attrs;
    size_t lineN = 0;
    XMLState state = XMLState::InFile;
    std::unique_ptr<XMLNode> xml;
    XMLNode* curNode = nullptr;
    XMLNode* prevNode = nullptr;

    while (std::getline(ifs, line))
    {
        size_t current_char_num = 0;
        ++lineN;

        while (current_char_num < line.length())
        {
            char c = line[current_char_num];

            switch (state)
            {
            case XMLState::InFile:
                switch (c)
                {
                case '<':
                    state = XMLState::InOpeningTag;
                    break;
                }

                break;

            case XMLState::InOpeningTag:
                switch (c)
                {
                case '>':
                    if (xml.get() == nullptr)
                    {
                        xml = std::make_unique<XMLNode>();
                        xml->name = tag;
                        xml->attributes = attrs;
                        curNode = xml.get();
                    }
                    else if (curNode != nullptr)
                    {
                        prevNode = curNode;
                        curNode->children.push_back(std::make_unique<XMLNode>());
                        curNode = curNode->children.back().get();
                        curNode->name = tag;
                        curNode->parent = prevNode;
                        curNode->attributes = attrs;
                    }

                    attrs.clear(); attr.clear();
                    tag.clear();
                    state = XMLState::InValue;
                    break;

                case '/':
                    state = XMLState::InClosingTag;
                    break;

                case '!':
                    if (tag.empty())
                    {
                        // ! is the immediate character after <
                        state = XMLState::InComment;
                    }

                    break;

                case ' ':
                    state = XMLState::InOpeningTagAttributes;
                    break;

                default:
                    tag += c;
                    break;
                }

                break;

            case XMLState::InClosingTag:
                switch (c)
                {
                case '>':
                    if (curNode->name != tag)
                    {
                        throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                            << "Closing tab </" << tag << "> on line " << lineN << " "
                            << "does not match with opening tag <" << curNode->name << ">").str()
                        );
                    }

                    curNode = curNode->parent;
                    tag.clear();
                    state = XMLState::InFile;

                    break;

                case ' ':
                    state = XMLState::InClosingTagAttributes;
                    break;

                default:
                    tag += c;
                    break;
                }

                break;

            case XMLState::InOpeningTagAttributes:
                switch (c)
                {
                case '>':
                    state = XMLState::InOpeningTag;
                    --current_char_num;
                    break;

                case ' ':
                    break;

                case '=':
                    state = XMLState::InOpeningTagAttribute;
                    break;

                default:
                    attr += c;
                    break;
                }

                break;

            case XMLState::InOpeningTagAttribute:
                switch (c)
                {
                case '"':
                    state = XMLState::InOpeningTagAttributeValue;
                    break;

                case ' ':
                    break;

                case '>':
                    attrs[attr] = value;
                    attr.clear();
                    value.clear();
                    state = XMLState::InOpeningTagAttributes;
                    --current_char_num;
                    break;

                default:
                    value += c;
                    break;
                }

                break;

            case XMLState::InOpeningTagAttributeValue:
                switch (c)
                {
                case '"':
                    attrs[attr] = value;
                    attr.clear();
                    value.clear();
                    state = XMLState::InOpeningTagAttributes;
                    break;

                default:
                    value += c;
                    break;
                }

                break;

            case XMLState::InClosingTagAttributes:
                switch (c)
                {
                case '>':
                    state = XMLState::InClosingTag;
                    --current_char_num;
                    break;
                }

                break;

            case XMLState::InValue:
                switch (c)
                {
                case '<':
                    curNode->value = trim(value);
                    value.clear();
                    state = XMLState::InOpeningTag;
                    break;

                default:
                    value += c;
                    break;
                }

                break;

            case XMLState::InComment:
                switch (c)
                {
                case '>':
                    state = XMLState::InValue;
                    break;
                }

                break;
            }

            ++current_char_num;
        }

        if (state == XMLState::InValue)
        {
            value += "\n";
        }
    }

    return xml;
}

}

#endif // YANNL_SIMPLE_XML_READER_H
