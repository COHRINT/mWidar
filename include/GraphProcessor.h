#ifndef GRAPH_PROCESSOR_H
#define GRAPH_PROCESSOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "dataStructures.h"
#include "Object.h"
#include <map>
#include <tuple>
#include <assert.h>
#define UPSCALE_FACTOR 4
#define IMAGE_SIZE 512 // size should be divisible by UPSCALE_FACTOR

using namespace cv;
class GraphProcessor
{
public:
    GraphProcessor();
    ~GraphProcessor();
    // void createProbabilityEllipse(); // use cv2 or other libraries to create an ellipse?
    cv::Mat processImage(const cv::Mat &image);
    Eigen::VectorXd scaleB(Eigen::VectorXd &b, int scale);
    cv::Mat updateMap(Eigen::VectorXd &b);
    void displayImage(const cv::Mat &image);
    cv::Mat drawCircle(cv::Mat &image, const cv::Point &center, int radius = 4, const cv::Scalar &color = cv::Scalar(255, 0, 217));
    cv::Mat drawSquare(cv::Mat &image, const cv::Point &center, int size = 4, const cv::Scalar &color = cv::Scalar(255, 0, 217));
    cv::Mat drawVelocityVector(cv::Mat &image, const cv::Point &center, const cv::Point &velocity, const cv::Scalar &color);
    cv::Mat writeTruthTargetsToImg(cv::Mat &image, std::vector<std::pair<int, cv::Point>> &targets);
    cv::Mat writeEstTargetsToImg(cv::Mat &image, std::vector<Object> &targets, bool drawVelocity = false);
    cv::Mat clearImage();
    int convertXYToBIndex(int x, int y);
    void resetB();
    void mapObjectToB(Eigen::VectorXd &b, Object obj);
    Eigen::VectorXd b; // probability vector
    cv::Mat img;
    cv::Mat originalImg;
};

#endif