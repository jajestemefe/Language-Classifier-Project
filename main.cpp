#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <valarray>
#include <algorithm>
#include <filesystem>
#include <random>

using namespace std;
namespace fs = filesystem;

auto getFrequencies(const string& text)-> vector<double>
{
    vector<double> frequencies(26, 0.);

    for (char c : text)
    {
        if (isalpha(c))
        {
            frequencies.at(tolower(c) - 'a')++;
        }
    }

    double length = 0.;
    for (double f : frequencies)
    {
        length += f * f;
    }
    length = sqrt(length);

    if (length > 0.)
    {
        for (double& f : frequencies)
        {
            f /= length;
        }
    }

    return frequencies;
}

struct Perceptron
{
    vector<double> weights;
    double threshold;
    double learningRate;

    explicit Perceptron(const vector<double>& w ={}, const double t = 0., const double lRate = 0.05)
    {
        weights = w;
        threshold = t;
        learningRate = lRate;
    }

    auto deltaRule(const double d, const double y, const vector<double>& inputs)-> void
    {
        if (weights.size() != inputs.size())
        {
            weights.resize(inputs.size(), 0.);
        }
        for (auto i = 0; i < inputs.size(); i++)
        {
            weights.at(i) += (d - y) * learningRate * inputs.at(i);
        }

        threshold -= (d - y) * learningRate;

        double norm = threshold * threshold;
        for (double w : weights) norm += w * w;
        norm = sqrt(norm);

        if (norm > 0.)
        {
            for (double& w : weights) w /= norm;
            threshold /= norm;
        }
    }

    [[nodiscard]] auto getOutput(const vector<double>& inputs) const -> double
    {
        if (weights.empty()) return 0.0;

        double y = 0.;
        double dotProduct = 0.;

        for (auto i = 0; i < weights.size(); i++)
        {
            dotProduct += weights.at(i) * inputs.at(i);
        }

        y = dotProduct - threshold;
        return y;
    }
};

struct Language
{
    string header;
    vector<vector<double>> fileFrequencies;

    explicit Language(const string& path)
    {
        const fs::path folder(path);
        header = folder.filename().string();

        for (const auto& entry : fs::directory_iterator(folder))
        {
            if (entry.is_regular_file())
            {
                ifstream file(entry.path());
                stringstream ss;
                ss << file.rdbuf();

                fileFrequencies.push_back(getFrequencies(ss.str()));
            }
        }
    }
};

struct TrainingSample
{
    int languageIndex;
    vector<double> frequencies;
};

auto main()-> int
{
    const string trainingPath = "../Languages";
    const string testingPath = "../Test";

    vector<Language> trainingLanguages;
    vector<Language> testingLanguages;


    if (fs::exists(trainingPath) && fs::is_directory(trainingPath))
    {
        cout << "Languages detected:" << endl;
        for (const auto& entry : fs::directory_iterator(trainingPath))
        {
            if (entry.is_directory())
            {
                trainingLanguages.emplace_back(entry.path().string());
                cout << "[" << trainingLanguages.back().header << "]"
                     << trainingLanguages.back().fileFrequencies.size() << " files" << endl;
            }
        }
    }

    if (trainingLanguages.empty())
    {
        cerr << "No languages to train!" << endl;
        return -1;
    }

    vector<TrainingSample> dataset;
    for (auto i = 0; i < trainingLanguages.size(); i++)
    {
        for (const auto& frequencies : trainingLanguages.at(i).fileFrequencies)
        {
            dataset.push_back({i, frequencies});
        }
    }

    vector<Perceptron> network(trainingLanguages.size(), Perceptron());

    cout << "Network training..." << endl;

    int maxEpochs = 1000;
    random_device rd;
    mt19937 g(rd());

    for (int epoch = 0; epoch < maxEpochs; epoch++)
    {
        ranges::shuffle(dataset, g);
        double totalError = 0.;

        for (const auto& sample : dataset)
        {
            for (auto i = 0; i < network.size(); i++)
            {
                double d = (sample.languageIndex == i) ? 1. : -1.;
                double y = network.at(i).getOutput(sample.frequencies);

                network.at(i).deltaRule(d, y, sample.frequencies);
                totalError += abs(d - y);
            }
        }

        if (totalError < 0.01)
        {
            cout << "Training is done after " << epoch << " epochs." << endl;
            break;
        }
    }

    if (fs::exists(testingPath) && fs::is_directory(testingPath))
    {
        for (const auto& entry : fs::directory_iterator(testingPath))
        {
            if (entry.is_directory())
            {
                testingLanguages.emplace_back(entry.path().string());
            }
        }
    }

    if (testingLanguages.empty())
    {
        cout << "No testing file" << endl;
    }
    else
    {
        cout << "\n--- Testing Performance ---" << endl;
        int totalCorrect = 0;
        int totalTests = 0;

        // Confusion matrix storage: [LanguageIndex][TP, FP, FN]
        // Index 0: True Positives, 1: False Positives, 2: False Negatives
        vector<vector<int>> metrics(trainingLanguages.size(), vector<int>(3, 0));

        for (auto i = 0; i < testingLanguages.size(); i++)
        {
            for (const auto& inputs : testingLanguages.at(i).fileFrequencies)
            {
                totalTests++;

                double maxOutput = -1e9;
                int predictedIndex = -1;

                for (auto j = 0; j < network.size(); j++)
                {
                    double y = network.at(j).getOutput(inputs);
                    if (y > maxOutput)
                    {
                        maxOutput = y;
                        predictedIndex = j;
                    }
                }

                if (predictedIndex == i)
                {
                    totalCorrect++;
                    metrics[i][0]++; // TP
                }
                else
                {
                    metrics[predictedIndex][1]++; // FP
                    metrics[i][2]++; // FN
                }
            }
        }

        double accuracy = static_cast<double>(totalCorrect) / totalTests;

        cout << "Accuracy: " << accuracy * 100. << "%\n" << endl;
        cout << left << setw(15) << "Language" << setw(12) << "Precision" << setw(10) << "Recall" << "F-Measure" << endl;
        cout << string(48, '-') << endl;

        for (auto i = 0; i < trainingLanguages.size(); i++)
        {
            double tp = metrics[i][0];
            double fp = metrics[i][1];
            double fn = metrics[i][2];

            double precision = (tp + fp > 0) ? (tp / (tp + fp)) : 0.;
            double recall = (tp + fn > 0) ?  (tp / (tp + fn)) : 0.;
            double fMeasure = (precision + recall > 0) ? (2 * precision * recall) / (precision + recall) : 0.;

            cout << left << setw(15) << trainingLanguages.at(i).header
                 << setw(12) << setprecision(3) << precision
                 << setw(10) << setprecision(3) << recall
                 << setprecision(3) << fMeasure << endl;
        }
    }

    cout << "\n--- Classification Ready ---" << endl;
    while (true)
    {
        cout << "\nEnter text to classify (or type 'stop' to quit): \n>>> ";
        string input;
        string line;

        while (getline(cin, line))
        {
            if (line == "stop")
            {
                input = line;
                break;
            }
            if (line.empty())
            {
                break;
            }
            input += line + " ";
        }
        if (input == "stop") break;
        if (input.empty() || input == " ") continue;

        vector<double> inputFrequencies = getFrequencies(input);

        double maxOutput = -1e9;
        int bestMatchIndex = -1;

        for (auto i = 0; i < network.size(); i++)
        {
            double y = network.at(i).getOutput(inputFrequencies);
            if (y > maxOutput)
            {
                maxOutput = y;
                bestMatchIndex = i;
            }
        }

        if (bestMatchIndex != -1)
        {
            cout << "Predicted language: " << trainingLanguages.at(bestMatchIndex).header << endl;
        }
    }

    return 0;
}