#include <cstdlib> // For std::system
#include "../../include/DataProcessor.h"
#include "../../include/GraphProcessor.h"
#include "../../include/VisionProcessor.h"
#include "../../include/Object.h"
#include "../../include/dataStructures.h"
#include <iostream>
#include <opencv2/opencv.hpp>
// #include <matio.h>
#include <filesystem>
#include <chrono>
#include <thread>

using namespace Eigen;
using namespace cv;
namespace fs = std::filesystem;

/*
General algorithm:
1. Read in the image using VP
2. Find the peaks in the image using peaks2 in VP
3. Create objects for each peak using the DP class
4. Propogate the state of the objects using a KF using DP
5. graph both the truth and KF estimations of the objects using the GP class
repeat 1-5
*/
int main(int argc, char *argv[])
{
    int counter = 0;
//    if (argc < 3)
//    {
//        std::cerr << "Usage: " << argv[0] << " <image_file> <truth_file>" << std::endl;
//        return -1;
//    }
//    std::system("python3 path/to/your/simulator.py");

    GraphProcessor gp = GraphProcessor();
    std::cout << "Running test of Kalman Filter" << std::endl;
    DataProcessor dp = DataProcessor();
    dp.printMatrices();
    VisionProcessor vp = VisionProcessor();
    // std::vector<std::pair<int, cv::Point>> oldTruth = vp.readTruthFile(argv[2]); // read initial truth data
    // create buffers for truth and image shared mem
    void *image_ptr = VisionProcessor::initializeConnectionToSharedMemory("image", V_IMAGE_SIZE);
    void *truth_ptr = VisionProcessor::initializeConnectionToSharedMemory("truth", V_OBJECT_SIZE * 2);
    sem_t *image_sem = VisionProcessor::initSemaphore("/image_sem", 0);
    sem_t *truth_sem = VisionProcessor::initSemaphore("/truth_sem", 0);
    while (true)
    {
        // openCV stuff
        char key = cv::waitKey(1);
        if (key == 'q')
        {
            break;
        }
        // read and find the peaks
        // std::pair<std::vector<cv::Point>, std::vector<std::pair<int, cv::Point>>> pairMapping = vp.getPeaksWithTruth(argv[1], argv[2]);
        std::vector<std::pair<int, cv::Point>> truthPeaksWithID = VisionProcessor::readDataAsVector(truth_ptr, truth_sem, 2);
        cv::Mat img = VisionProcessor::readDataAsImage(image_ptr, image_sem, IMAGE_SIZE);
        std::vector<cv::Point> allImgPeaks = VisionProcessor::findPeaks(img, 0.9);
        std::vector<std::pair<cv::Point, Object *>> truthObjectMap = dp.truthDataMapping(allImgPeaks, truthPeaksWithID); // one truth per object
        for (auto pairing : truthObjectMap)
        {
            Eigen::VectorXd measurement(2);
            measurement << pairing.first.x, pairing.first.y;
            dp.propogateState(*pairing.second, measurement, "kalman-filter");
            // std::cout << "xplus: " << pairing.second.getStateVector() << std::endl;
            // std::cout << "pplus: " << pairing.second.getStateCovariance() << std::endl;
            // gp.mapObjectToB(gp.b, pairing.second);
        }

        // graph the truth and KF estimations of the objects
        std::vector<Object> objects = dp.getObjects();
        for (auto &obj : objects)
        {
            std::cout << "obj " << obj.getID() << " state:\n"
                      << obj.getStateVector() << std::endl;
            std::cout << "obj " << obj.getID() << " measurement vector:\n"
                      << std::endl;
            for (int i = 0; i < obj.getMeasurementState().size(); i++)
            {
                std::cout << obj.getMeasurementState()(i) << std::endl;
            }
        }

        // gp.img = gp.clearImage();
        std::cout << "test" << counter << std::endl;
        counter++;
        gp.img = gp.updateMap(gp.b); // idk
        gp.img = gp.writeEstTargetsToImg(gp.img, objects, true);
        gp.img = gp.writeTruthTargetsToImg(gp.img, truthPeaksWithID);

        // show the image
        cv::imshow("Colored Matrix", gp.img);
        // vp.readAndDisplayRawImg(argv[1]);
    }
}