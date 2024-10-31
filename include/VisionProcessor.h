#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> // Include for cv::imread
#include <sys/mman.h>
#include <semaphore.h>

using namespace cv;

class VisionProcessor
{
public:
    struct peakTruthMapping
    {
        cv::Point peak;
        int value;
    };
    void *createImageBuffer(const char *filename, int width, int height);
    void *createTruthBuffer(const char *filename, int num_truth_points);
    void *getImageBuffer();
    void *getTruthBuffer();
    static std::vector<cv::Point> findPeaks(const cv::Mat &image, double threshold);
    static std::vector<cv::Point> openAndRead(const char *filename);
    static std::vector<cv::Point> readSharedBuffer(const char *filename);
    cv::Mat readSharedImgBuffer(void *shared_mem_ptr, int width, int height);
    std::vector<std::pair<int, cv::Point>> readSharedTruthBuffer(void *shared_mem_ptr, int num_truth_points);
    static void readAndDisplayRawImg(const char *filename);
    static int exportValsAsCSV(const std::string &filename, const std::string &pixelVals);
    static bool peaksWithinRange(std::vector<cv::Point> peaks, int x, int y, int range);
    std::pair<std::vector<cv::Point>, std::vector<std::pair<int, cv::Point>>> getPeaksWithTruth(const char *filename, const char *truthFileJson);
    std::vector<std::pair<int, cv::Point>> readTruthFile(const char *filename);

private:
    void *image_ptr;
    void *truth_ptr;
    bool is_image_buffer_empty = true;
    bool is_truth_buffer_empty = true;
};