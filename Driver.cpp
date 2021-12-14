#pragma once

#include <iostream>
#include <chrono>
#include "Neuranet/Math/headers/Matrix2D.hpp"
#include "Neuranet/Math/headers/Matrix3D.hpp"
#include "Neuranet/Math/headers/Polynomial.hpp"
#include "Neuranet/Networks/Neural/headers/NeuralNetwork.hpp"
#include "Neuranet/Util/headers/Dataset.hpp"
#include "Neuranet/Util/headers/Image.hpp"

using namespace std;
using namespace Neuranet;

int main()
{
    try
    {
        auto startTime = std::chrono::steady_clock::now();

        std::vector<uint16_t> nodeCounts = { 3, 1, 3 };
        NeuralNetwork net = NeuralNetwork(nodeCounts, Activation::SIGMOID);
        cout << net << endl;

        double* vals = new double[] { -0.4, 0.7, 0.9 };
        Matrix2D input = Matrix2D(3, 1, vals);
        delete[] vals;

        cout << "Input: " << input << endl;
        Matrix2D output = net.compute(input);
        cout << "Output: " << output << endl;

        std::vector<Matrix2D> ins;
        std::vector<Matrix2D> outs;
        std::string inName = "C:/Users/scrat/Documents/GitHub/Tests/NeuranetCPP/Datasets/d3ins.csv";
        std::string outName = "C:/Users/scrat/Documents/GitHub/Tests/NeuranetCPP/Datasets/d3expouts.csv";
        inName = "C:/Users/scrat/Documents/GitHub/Tests/NeuranetCPP/Datasets/datasets1Ins.csv";
        outName = "C:/Users/scrat/Documents/GitHub/Tests/NeuranetCPP/Datasets/datasets1ExpOuts.csv";
        Dataset::parse(inName, outName, ins, outs);

        cout << "AVERAGE LOSS BEFORE LEARNING:\n\t" << net.getAverageLoss(ins, outs) << endl;

        net.learn(inName, outName, 1500, 3, 3.0);
        //inName = "C:/Users/scrat/Documents/GitHub/Tests/NeuranetCPP/datasets1Ins.csv";
        //outName = "C:/Users/scrat/Documents/GitHub/Tests/NeuranetCPP/datasets1ExpOuts.csv";
        //net.learn(inName, outName, 1500, 1, 3.0);

        cout << net << endl;

        cout << "AVERAGE LOSS AFTER LEARNING:\n\t" << net.getAverageLoss(ins, outs) << endl;

        cout << net.compute(input) << endl;

        std::string folPath = "C:/Users/scrat/Documents/GitHub/Tests/NeuranetCPP/WeightsAndBiases";
        net.writeToPath(folPath);

        double* vv = new double[] { 2, 2, 2 };
        Matrix2D in(3, 1, vv);

        cout << net.compute(in) << endl;

        uint16_t correct = 0;
        for (uint16_t i = 0; i < ins.size(); i += 1)
        {
            if (outs[i].getIndexOfMax() == net.compute(ins[i]).getIndexOfMax())
            {
                correct += 1;
            }
        }

        cout << "Accuracy: " << std::to_string(100.0 * correct / ins.size()) << "% (" << correct << "/" << ins.size() << ")" << endl;

        delete[] vv;

        cout << "TIME: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTime).count() * 0.000000001) << endl;
    }
    catch (std::string e)
    {
        cout << e << endl;
    }

    return 0;
}