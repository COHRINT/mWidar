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
#include <csignal>

using namespace Eigen;
using namespace cv;
namespace fs = std::filesystem;

sem_t *image_sem;
sem_t *truth_sem;
void *image_ptr;
void *truth_ptr;

void signalHandler(int signum)
{
    std::cout << "Interrupt signal (" << signum << ") received. Cleaning up.\n";
    sem_close(image_sem);
    sem_close(truth_sem);
    sem_unlink("/image_sem");
    sem_unlink("/object_sem");
    exit(signum);
}


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
    signal(SIGTERM, signalHandler);
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
    image_sem = VisionProcessor::initSemaphore("/image_sem", 0);
    truth_sem = VisionProcessor::initSemaphore("/object_sem", 0);
    image_ptr = VisionProcessor::initializeConnectionToSharedMemory("/image", V_IMAGE_SIZE);
    truth_ptr = VisionProcessor::initializeConnectionToSharedMemory("/objects", V_OBJECT_SIZE * 2);
    while (true)
    {
        // openCV stuff
        char key = cv::waitKey(1);
        if (key == 'q')
        {
            break;
        }
        // read and find the peaks (in mWidar frame)
        // std::pair<std::vector<cv::Point>, std::vector<std::pair<int, cv::Point>>> pairMapping = vp.getPeaksWithTruth(argv[1], argv[2]);
        std::vector<std::pair<int, cv::Point>> truthPeaksWithID = VisionProcessor::readDataAsVector(truth_ptr, truth_sem, 1);
        cv::Mat img = VisionProcessor::readDataAsImage(image_ptr, image_sem, 128);
        // std::cout << "truthPeaksWithID: " << std::endl;
        for (auto &pair : truthPeaksWithID)
        {
            std::cout << "Object " << pair.first << " at: " << pair.second << std::endl;
        }
        std::vector<cv::Point> allImgPeaks = VisionProcessor::findPeaks(img, 0.75); // finds peaks in CV frame
        for (auto &peak : allImgPeaks)
        {
            DataProcessor::convertCoordinateFrame(peak, "mWidar");
        }
        std::vector<std::pair<cv::Point, Object *>> truthObjectMap = dp.truthDataMapping(allImgPeaks, truthPeaksWithID); // one truth per object
        for (auto pairing : truthObjectMap)
        {
            Eigen::VectorXd measurement(2); // Initialize with size 2
            measurement << pairing.first.x, pairing.first.y; // Assign values
            dp.propogateState(*pairing.second, measurement, "kalman-filter"); // propogateState(Object, Point, filter)
            // std::cout << "xplus: " << pairing.second->getStateVector() << std::endl;
            // std::cout << "z: " << pairing.second->getMeasurementState() << std::endl;
            // std::cout << "pplus: " << pairing.second.getStateCovariance() << std::endl;
            // gp.mapObjectToB(gp.b, pairing.second);
        }

        // graph the truth and KF estimations of the objects
        std::vector<Object> objects = dp.getObjects();
        // for (auto &obj : objects)
        // {
        // std::cout << "objects size: " << objects.size() << std::endl;
        //     std::cout << "obj " << obj.getID() << " state:\n"
        //               << obj.getStateVector() << std::endl;
        //     std::cout << "obj " << obj.getID() << " measurement:"
        //               << std::endl;
        //     for (int i = 0; i < obj.getMeasurementState().size(); i++)
        //     {
        //         std::cout << obj.getMeasurementState()(i) << std::endl;
        //     }
        // }

        // gp.img = gp.clearImage();

        // TODO: convert truth targets back into openCV frame
        // TODO: convert KF targets back into openCV frame
        // std::cout << "test" << counter << std::endl;
        counter++;

        std::vector<cv::Point> truthcvpoints;
        truthcvpoints.reserve(truthPeaksWithID.size());
        for (auto &point : truthPeaksWithID)
        {
            DataProcessor::convertCoordinateFrame(point.second, "openCV");
            truthcvpoints.push_back(point.second);
        }
        gp.img = gp.updateMap(gp.b); // idk
        gp.img = gp.writeGeneralPoint(gp.img, allImgPeaks); // draw the peaks
        gp.img = gp.writeEstTargetsToImg(gp.img, objects, true);
        gp.img = gp.writeTruthTargetsToImg(gp.img, truthcvpoints);
        truthcvpoints.clear();
        // show the image
        cv::imshow("Input IMG", img);
        cv::imshow("Colored Matrix", gp.img);
        // vp.readAndDisplayRawImg(argv[1]);
    }
}