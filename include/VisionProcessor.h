#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <semaphore.h>
#include <sys/mman.h>
#define V_IMAGE_SIZE 128 * 128
#define V_OBJECT_SIZE 3 * 4

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
    static cv::Mat readDataAsImage(void* shared_mem_ptr, sem_t* semaphore, int size);
    static std::vector<cv::Point> findPeaks(const cv::Mat &image, double threshold);
    static std::vector<cv::Point> openAndRead(const char *filename);
    static std::string readDataAsString(void* shared_mem_ptr, sem_t* semaphore, int num_objects);
    static std::vector<std::pair<int, cv::Point>> readDataAsVector(void* shared_mem_ptr, sem_t* semaphore, int num_objects);
    static sem_t* initSemaphore(const char* name, int value);
    static void* initializeConnectionToSharedMemory(const char* name, int size);
    static void readAndDisplayRawImg(const char *filename);
    static int exportValsAsCSV(const std::string &filename, const std::string &pixelVals);
    static bool peaksWithinRange(const std::vector<cv::Point>& peaks, int x, int y, int range);
    std::pair<std::vector<cv::Point>, std::vector<std::pair<int, cv::Point>>> getPeaksWithTruth(const char *filename, const char *truthFileJson);
    std::vector<std::pair<int, cv::Point>> readTruthFile(const char *filename);
};