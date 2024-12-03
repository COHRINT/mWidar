/*
mWidar Simulator -- Simulator.hpp

*/

#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>


// Define the class Simulator

class Simulator {
public:
    // Constructor
    Simulator(std::string filename);

    // Destructor
    ~Simulator();

    // Public member functions
    int run();
    int returnSim();

private:
    // Private member functions
    void loadModel(std::string filepath);
    void readData();
    void calculateResults();

    // Private member variables
    std::string filepath;
    // Model Matricies (R and S)
    // std::vector<std::vector<double>> R;
    // std::vector<std::vector<double>> S;
};


