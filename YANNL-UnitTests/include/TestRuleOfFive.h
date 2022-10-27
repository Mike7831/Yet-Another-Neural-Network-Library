#ifndef TEST_RULE_OF_FIVE_H
#define TEST_RULE_OF_FIVE_H

#include <iostream>
#include <vector>

class TestRuleOfFive
{
public:
    TestRuleOfFive()
        : p{ new int{std::rand()} }
    {
        std::cerr << "TestRuleOfFive() " << *p << std::endl;
    }

    TestRuleOfFive(const TestRuleOfFive& other)
        : p{ new int{*(other.p)} }
    {
        std::cerr << "TestRuleOfFive(const TestRuleOfFive& other)" << std::endl;
    }

    TestRuleOfFive(TestRuleOfFive&& other) noexcept
        : p{ other.p }
    {
        other.p = nullptr;
        std::cerr << "TestRuleOfFive(TestRuleOfFive&& other)" << std::endl;
    }

    TestRuleOfFive& operator=(const TestRuleOfFive& other)
    {
        if (&other != this)
        {
            delete p;
            p = nullptr;
            p = new int{ *(other.p) };
        }
        std::cerr << "TestRuleOfFive& operator=(const TestRuleOfFive& other)" << std::endl;
        return *this;
    }

    TestRuleOfFive& operator=(TestRuleOfFive&& other) noexcept
    {
        if (&other != this)
        {
            delete p;
            p = other.p;
            other.p = nullptr;
        }
        std::cerr << "TestRuleOfFive& operator=(TestRuleOfFive&& other)" << std::endl;
        return *this;
    }

    ~TestRuleOfFive()
    {
        delete p;
        std::cerr << "~TestRuleOfFive() " << p << std::endl;
    }

    static std::vector<TestRuleOfFive> generateVect(const std::vector<TestRuleOfFive>& in)
    {
        std::vector<TestRuleOfFive> out;
        out.push_back(TestRuleOfFive());
        return out;
    }

    static void testWithMove()
    {
        std::vector<TestRuleOfFive> v;
        v.push_back(TestRuleOfFive());

        for (size_t n = 0; n < 2; n++) { v = std::move(TestRuleOfFive::generateVect(v)); }

        std::cerr << std::endl;
    }

    static void testWithoutMove()
    {
        std::vector<TestRuleOfFive> v;
        v.push_back(TestRuleOfFive());

        for (size_t n = 0; n < 2; n++) { v = TestRuleOfFive::generateVect(v); }

        std::cerr << std::endl;
    }

private:
    int* p;
};

#endif // TEST_RULE_OF_FIVE_H
