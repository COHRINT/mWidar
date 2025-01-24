#pragma once
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <matio.h>
#include "Object.h"
#include <string>

using namespace cv;
template <typename eig_t>
    static eig_t scale(eig_t &matrix_or_vector, double scale)
{
    matrix_or_vector *= scale;
}

class DataProcessor
/*
This class is responsible for processing the data from the vision processor and updating the state of the existing objects.
*/
{
public:
    DataProcessor();                                  // Default constructor
    DataProcessor(int imageSize, std::string filter); // Constructor with image size
    ~DataProcessor();                                 // Default destructor
    // Object propogateState(Object *obj, std::string filter = NULL);                                            // Update the state and cov based on a new point (data from vision processor
    // std::map<DataStructures::point, Object> associatePeaks(cv::set<Object> peaks, std::set<Object> &objects); // Associate peaks with objects
    // float **createUncertaintyBounds(Object *obj);                                                             // Returns 2D array containing the probability of the object being at a certain point
    // int calculateError(Object *obj, DataStructures::point);                                                   // Calculate the error in pixels between the predicted and actual position
    void initializeStateMatrices(); // Initialize the state matrices for the objects
    std::vector<Object> getObjects();
    Eigen::SparseMatrix<double> createBlockMatrix(const Eigen::MatrixXd &A,
                                                  const Eigen::MatrixXd &Gamma,
                                                  const Eigen::MatrixXd &W);
    static void processMatFile(mat_t *matfp, matvar_t *matvar, Eigen::SparseMatrix<double> *sparseMat);
    static void convertMatFileToEigenSparseMatrix(const char *fileName, Eigen::SparseMatrix<double> *sparseMat);
    std::vector<Object> createObjectsFromPeaks(std::vector<cv::Point> &peaks);
    std::vector<std::pair<cv::Point, Object>> nearestNeighborMapping(std::vector<cv::Point> &measurements);
    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> Gamma;
    Eigen::SparseMatrix<double> H;
    Eigen::SparseMatrix<double> W;
    Eigen::SparseMatrix<double> R;
    Eigen::SparseMatrix<double> F;
    Eigen::SparseMatrix<double> Z;
    Eigen::SparseMatrix<double> eZ;
    Eigen::SparseMatrix<double> Q;
    Eigen::SparseMatrix<double> manualMatrixExponential(Eigen::MatrixXd &A, int terms);

    void propogateState(Object &obj, Eigen::VectorXd &measurement, const std::string& filter);
    void markovUpdate(Object &obj, Eigen::VectorXd &measurement);
    void kalmanUpdate(Object &obj, Eigen::VectorXd &measurement);
    void particleUpdate(Object &obj, Eigen::VectorXd &measurement);
    void prettyPrintObjectData();
    std::vector<std::pair<cv::Point, Object *>> truthDataMapping(std::vector<cv::Point> &measurements, std::vector<std::pair<int, cv::Point>> &truthData);

    template<class eig_t>
    static void convertMetersToPx(eig_t &eigen_type, double scale);
    static void convertCoordinateFrame(Eigen::VectorXd &coordinate_vector, const std::string &new_frame);
    static void convertCoordinateFrame(cv::Point &coordinate_point, const std::string &new_frame);
    static std::pair<cv::Point, std::pair<int, cv::Point>> findClosestPointToTruth(std::vector<cv::Point> &measurements, std::pair<int, cv::Point> &truthPoint);
    void printMatrices();

    // void createStateMatrices();                                                                               // Create the state matrices for some object
private:
    // Eigen::MatrixXd observationMatrix; // Observation matrix O
    Eigen::MatrixXd observationNoise; // Observation noise R
    Eigen::MatrixXd kalmanGain;       // Kalman gain K
    int timestep;                     // Timestep for data propogation
    void markovChain();               // Markov chain to propogate the state of the object
    void kalmanFilter();              // Kalman filter to propogate the state of the object
    void particleFilter();            // Particle filter to propogate the state of the object
    std::vector<Object> objects;
};