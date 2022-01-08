#pragma once

#include <iostream>
#include <chrono>
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include "Neuranet/Math/headers/Matrix2D.hpp"
#include "Neuranet/Math/headers/Matrix3D.hpp"
#include "Neuranet/Math/headers/Polynomial.hpp"
#include "Neuranet/Networks/Neural/headers/NeuralNetwork.hpp"
#include "Neuranet/Networks/CNN/headers/CNN.hpp"
#include "Neuranet/Util/headers/Dataset.hpp"
#include "Neuranet/Util/headers/Image.hpp"

using namespace std;
using namespace Neuranet;

int main()
{
    try
    {
        auto startTime = std::chrono::steady_clock::now();

        /*std::vector<uint16_t> nodeCounts = {3, 15, 15, 3};
        NeuralNetwork net = NeuralNetwork(nodeCounts);
        std::vector<Matrix2D> inputs;
        std::vector<Matrix2D> outputs;
        std::string inputFileName = "C:\\Users\\scrat\\OneDrive\\Documents\\Dev\\C++ Projects\\NeuranetCPP\\Datasets\\datasets1Ins.csv";
        std::string outputFileName = "C:\\Users\\scrat\\OneDrive\\Documents\\Dev\\C++ Projects\\NeuranetCPP\\Datasets\\datasets1ExpOuts.csv";
        Dataset::parse(inputFileName, outputFileName, inputs, outputs);
        
        cout << net << endl;

        cout << "Average Loss Before Learning: " << net.getAverageLoss(inputs, outputs) << endl;
        net.learn(inputs, outputs, 1500, 1, 1.0);
        cout << "Average Loss After Learning: " << std::to_string(net.getAverageLoss(inputs, outputs)) << endl;

        int correct = 0;
        for (int index = 0; index < inputs.size(); index += 1)
        {
            int guess = net.compute(inputs[index]).getIndexOfMax();
            int answer = outputs[index].getIndexOfMax();
            if (guess == answer)
            {
                correct += 1;
            }
            else
            {
                cout << "\nInput:" << inputs[index] << endl;
                cout << "Output:" << net.compute(inputs[index]) << endl;
                cout << "Expected Output:" << outputs[index] << endl;
            }
        }

        cout << "Accuracy: " << (100.0 * correct / inputs.size()) << "%" << endl;

        cout << net << endl;

        std::string wabFile = "C:\\Users\\scrat\\OneDrive\\Documents\\Dev\\C++ Projects\\NeuranetCPP\\WeightsAndBiases";
        net.writeToPath(wabFile);*/

        std::vector<Matrix3D> images;
        std::vector<Matrix2D> labels;
        std::string imagesFile = "C:\\Users\\scrat\\OneDrive\\Documents\\Dev\\C++ Projects\\NeuranetCPP\\Datasets\\train-images.idx3-ubyte";
        std::string labelsFile = "C:\\Users\\scrat\\OneDrive\\Documents\\Dev\\C++ Projects\\NeuranetCPP\\Datasets\\train-labels.idx1-ubyte";
        Dataset::parseMNISTimages(imagesFile, labelsFile, images, labels);

        CNN cnn = CNN(10, Activation::SOFTMAX);

        /*double* vals = new double[]
        {
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0
        };
        Matrix3D fim = Matrix3D(5, 5, 1, vals);
        images = { fim };
        delete[] vals;

        double* vals2 = new double[]
        {
            1, 0, 0
        };
        Matrix2D fla = Matrix2D(3, 1, vals2);
        labels = { fla };
        delete[] vals2;*/
        
        ConvolutionalLayer layOne = ConvolutionalLayer(8, 5, 5, 1, std::make_tuple<uint8_t, uint8_t>(1, 1), 0, Activation::SIGMOID);
        cnn.addConvolution(&layOne);
        //PoolingLayer layTwo = PoolingLayer(std::make_tuple<uint8_t, uint8_t>(3, 3), std::make_tuple<uint8_t, uint8_t>(3, 3), Pooling::MAX);
        //cnn.addConvolution(&layTwo);
        //ConvolutionalLayer layThree = ConvolutionalLayer(2, 3, 3, 4, std::make_tuple<uint8_t, uint8_t>(1, 1), 1, Activation::RELU);
        //cnn.addConvolution(&layThree);
        //PoolingLayer layFour = PoolingLayer(std::make_tuple<uint8_t, uint8_t>(3, 3), std::make_tuple<uint8_t, uint8_t>(3, 3), Pooling::MAX);
        //cnn.addConvolution(&layFour);
        //ConvolutionalLayer layFive = ConvolutionalLayer(2, 3, 3, 4, std::make_tuple<uint8_t, uint8_t>(1, 1), 1, Activation::SIGMOID);
        //cnn.addConvolution(&layFive);
        cnn.addHiddenLayer(60, Activation::SIGMOID);
        //cnn.addHiddenLayer(60, Activation::RELU);
        //cnn.addHiddenLayer(16, Activation::SIGMOID);


        cnn.compute(images[0]);
        //cout << cnn << endl;

        cout << "\nGradients: " << endl;
        std::vector<std::vector<Matrix3D>> filterWeightGradients;
        std::vector<std::vector<Matrix2D>> filterBiasGradients;
        cnn.getGradients(images[0], labels[0], filterWeightGradients, filterBiasGradients);
        //cout << (filterWeightGradients[filterWeightGradients.size() - 1])[filterWeightGradients[filterWeightGradients.size() - 1].size() - 1] << endl;
        cout << (filterBiasGradients[filterBiasGradients.size() - 1])[filterBiasGradients[filterBiasGradients.size() - 1].size() - 1] << endl;

        //cout << "Input: " << images[0] << endl;
        cout << "Output: " << cnn.compute(images[0]) << endl;
        cout << "Expected Output: " << labels[0] << endl;

        std::vector<Matrix3D>::const_iterator first = images.begin();
        std::vector<Matrix3D>::const_iterator last = images.begin() + 6000;
        std::vector<Matrix3D> newImages(first, last);

        std::vector<Matrix2D>::const_iterator first2 = labels.begin();
        std::vector<Matrix2D>::const_iterator last2 = labels.begin() + 6000;
        std::vector<Matrix2D> newLabels(first2, last2);

        double avgLoss = cnn.getAverageLoss(newImages, newLabels);
        cout << "Average loss before learning: " << avgLoss << endl;
        cnn.learn(newImages, newLabels, 1, 5, 0.1);
        avgLoss = cnn.getAverageLoss(newImages, newLabels);

        cout << cnn << endl;

        int correct = 0;
        for (int i = 0; i < 600; i += 1)
        {
            Matrix2D out = cnn.compute(images[i]);
            Matrix2D actualOut = labels[i];
            if (out.getIndexOfMax() == actualOut.getIndexOfMax())
            {
                correct += 1;
            }
        }

        cout << cnn.compute(images[0]) << endl;

        cout << "Average loss after learning: " << avgLoss << endl;
        cout << "Accuracy on first 600: " << (100.0 * correct / 600) << "% (" << correct << " / " << "600)" << endl;

        cout << "TIME: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTime).count() * 0.000000001) << endl;
    }
    catch (std::string e)
    {
        cout << e << endl;
    }

    _CrtDumpMemoryLeaks();
    return 0;
}