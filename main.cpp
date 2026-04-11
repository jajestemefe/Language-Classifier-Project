#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <valarray>
#include <algorithm>
#include <map>

using namespace std;

struct Perceptron
{
    vector<double> weights;
    double threshold;
    double learningRate;

    explicit Perceptron(const vector<double>& w ={}, const double t = 0., const double lRate = 1.)
    {
        weights = w;
        threshold = t;
        learningRate = lRate;
    }

    auto deltaRule(const double d, const double y, const vector<double>& inputs)-> void
    {
        if (weights.size() != inputs.size())
        {
            weights.resize(inputs.size());
        }
        for (auto i = 0; i < inputs.size(); i++)
        {
            weights.at(i) += (d - y) * learningRate * inputs.at(i);
        }
    }

    [[nodiscard]] auto getOutput(const vector<double>& inputs) const -> double
    {
        double y = 0.;
        double dotProduct = 0.;

        for (auto i = 0; i < weights.size(); i++)
        {
            dotProduct = weights.at(i) * inputs.at(i);
        }

        y = dotProduct - threshold;
        return y;
    }

};

auto main()-> int
{
    return 0;
}