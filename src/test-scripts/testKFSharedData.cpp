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

    // initialize the classes
    GraphProcessor gp = GraphProcessor();
    std::cout << "Running test of Kalman Filter" << std::endl;
    DataProcessor dp = DataProcessor();
    dp.printMatrices();
    VisionProcessor vp = VisionProcessor();


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
        // read in the image and the truth data (truth data is in mWidar frame)
        std::vector<std::pair<int, cv::Point>> truthPeaksWithID = VisionProcessor::readDataAsVector(truth_ptr, truth_sem, 1); // get truth peaks in mWidar frame
        cv::Mat img = VisionProcessor::readDataAsImage(image_ptr, image_sem, 128);
        for (auto &pair : truthPeaksWithID)
        {
            std::cout << "Object " << pair.first << " at: " << pair.second << std::endl;
        }

        // find peaks and convert them to the mWidar frame for KF
        std::vector<cv::Point> allImgPeaks = VisionProcessor::findPeaks(img, 0.85); // finds peaks in CV frame
        std::vector<cv::Point> allImgPeaksMwidar = allImgPeaks;
        for (auto &peak : allImgPeaksMwidar)
        {
            DataProcessor::convertCoordinateFrame(peak, "mWidar");
        }

        // map the closest peak to the truth data
        std::vector<std::pair<cv::Point, Object *>> truthObjectMap = dp.truthDataMapping(allImgPeaksMwidar, truthPeaksWithID); // one truth per object

        // using the closest peak to the truth data, propagate the state of the objects using a KF
        for (auto pairing : truthObjectMap)
        {
            Eigen::VectorXd measurement(2); // Initialize with size 2
            measurement << pairing.first.x, pairing.first.y; // Assign values
            // std::cout << "truth:       (" << pairing.first.x << ", " << pairing.first.y << ")" << std::endl;
            // std::cout << "object:      (" << pairing.second->getStateVector()(0) << ", " << pairing.second->getStateVector()(2) << ")" << std::endl;

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
        //     std::cout << "obj " << obj.getID() << " state:\n" << obj.getStateVector() << std::endl;
        //     std::cout << "obj " << obj.getID() << " measurement:" << std::endl;
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
        gp.img = gp.updateMap(gp.b); // for probability heatmap
        // gp.img = gp.writeGeneralPoint(gp.img, allImgPeaks); // draw the peaks
        gp.img = gp.writeEstTargetsToImg(gp.img, objects, true);
        gp.img = gp.writeTruthTargetsToImg(gp.img, truthcvpoints);
        truthcvpoints.clear();
        // show the image
        cv::imshow("Input IMG", img);
        cv::imshow("Colored Matrix", gp.img);
        // vp.readAndDisplayRawImg(argv[1]);
    }
}