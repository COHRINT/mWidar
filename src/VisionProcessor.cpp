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
#include "../include/json.hpp"
#include "../include/VisionProcessor.h"
#include <thread>

#ifdef _WIN32
#include <windows.h>
#define WINDOWS 1
#define UNIX 0
#elif defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#define WINDOWS 0
#define UNIX 1
#else
#error "Unknown operating system"
#endif

#define BOTTOM_THRESHOLD 128 - 25
using namespace cv;
using json = nlohmann::json;

/**
 * @brief Find local peaks in an image.
 *
 * This function finds local peaks in an image by comparing each pixel to its 8 neighbors.
 * A pixel is considered git gita peak if it is greater than all of its neighbors.
 *
 *
 * @param image The input image.
 * @param threshold The threshold value for a pixel to be considered a peak. Default is 0.90.
 * @return A vector of points representing the locations of the peaks.
 */
std::vector<cv::Point> VisionProcessor::findPeaks(const cv::Mat &image, double threshold = 0.40)
{
    std::vector<cv::Point> peaks;

    // Find the maximum pixel value in the image
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);

    // Calculate the threshold as 90% of the maximum pixel value
    threshold = static_cast<float>(threshold * maxVal);

    // Iterate through each pixel in the image, excluding the border pixels
    for (int y = 1; y < image.rows - 1; ++y)
    {
        for (int x = 1; x < image.cols - 1; ++x)
        {
            // Get the value of the current pixel
            int currentValue = static_cast<int>(image.at<uchar>(y, x));
            if (currentValue < threshold)
                continue; // Skip pixels below the threshold

            // Check the 8 neighbors to see if the current pixel is a local maximum
            bool isPeak = true;
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0)
                        continue; // Skip the current pixel itself

                    if (static_cast<int>(image.at<uchar>(y + dy, x + dx)) > currentValue)
                    {
                        isPeak = false;
                        break;
                    }
                }
                if (!isPeak)
                    break;
            }

            // If the current pixel is a local maximum, add it to the peaks vector
            if (isPeak && !(peaksWithinRange(peaks, x, y, 2)) && y < BOTTOM_THRESHOLD) // and if there are no other peaks within x pixels
            {
                cv::Point newPeak = cv::Point(x, y);
                peaks.push_back(newPeak);
            }
        }
    }

    return peaks;
}

bool VisionProcessor::peaksWithinRange(const std::vector<cv::Point>& peaks, const int x, const int y, const int range)
{
    for (const auto& peak : peaks)
    {
        if (cv::norm(peak.x - x) < range || cv::norm(peak.y - y) < range)
        {
            return true;
        }
    }
    return false;
}

cv::Mat VisionProcessor::readDataAsImage(void *shared_mem_ptr, sem_t *semaphore, int size)
{
    sem_wait(semaphore);
    // Ensure the shared memory pointer is not null
    cv::Mat image(128, 128, CV_8UC4, shared_mem_ptr);
    cv::Mat img_cpy = image.clone();
    cv::normalize(img_cpy, img_cpy, 0, 255, cv::NORM_MINMAX);
    cv::cvtColor(img_cpy, img_cpy, cv::COLOR_BGRA2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(15.0); // Adjust the clip limit to control the contrast enhancement
    clahe->apply(img_cpy, img_cpy);
    return img_cpy;
}

std::string VisionProcessor::readDataAsString(void *shared_mem_ptr, sem_t *semaphore, const int num_objects)
{
    sem_wait(semaphore);
    std::string result;
    int *data = static_cast<int *>(shared_mem_ptr);
    for (size_t i = 0; i < num_objects; ++i)
    {
        int x = data[i * 3];
        int y = data[i * 3 + 1];
        int counter = data[i * 3 + 2];
        result += "Object " + std::to_string(i) + ": x=" + std::to_string(x) + ", y=" + std::to_string(y) + ", counter=" + std::to_string(counter) + "\n";
    }
    return result;
}

std::vector<std::pair<int, cv::Point>> VisionProcessor:: readDataAsVector(void *shared_mem_ptr, sem_t *semaphore, const int num_objects)
{
    sem_wait(semaphore);
    std::vector<std::pair<int, cv::Point>> result;
    auto data = static_cast<int *>(shared_mem_ptr);
    for (size_t i = 0; i < num_objects; ++i)
    {
        int x = data[i * 3];
        int y = data[i * 3 + 1];
        // int counter = data[i * 3 + 2];
        result.emplace_back(i, cv::Point(x, y));
    }
    return result;
}

sem_t *VisionProcessor::initSemaphore(const char *name, const int value)
{
    sem_t *sem = sem_open(name, O_CREAT, 0666, value);
    if (sem == SEM_FAILED)
    {
        std::cerr << "Failed to create semaphore" << std::endl;
        return nullptr;
    }
    return sem;
}

void *VisionProcessor::initializeConnectionToSharedMemory(const char *name, const int size)
{
    // Open the shared memory segment
    int shm_fd;
    while (shm_fd = shm_open(name, O_RDONLY, 0666), shm_fd == -1)
    {
        std::cout << "Waiting for shared mem to open" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    // Map the shared memory segment to the process's address space
    void *shared_mem_ptr = mmap(nullptr, size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shared_mem_ptr == MAP_FAILED)
    {
        std::cerr << "Failed to map shared memory" << std::endl;
        close(shm_fd);
        return nullptr;
    }

    std::cout << "Shared memory initialized" << std::endl;
    close(shm_fd); // Close the file descriptor as it is no longer needed after mmap
    return shared_mem_ptr;
}
/**
 * @brief Open the image file and load the data, locking the file from other processes.
 *
 * This function opens the file and loads the data, locking the file from other processes.
 * If the file is being written to by another process, the function will wait until the file is available.
 * This makes use of the fcntl system call to lock the file for unix, and the LockFileEx function for windows.
 *
 * @param filename The name of the file to open and read.
 * @return the set of peaks found in the image.
 */
std::vector<cv::Point> VisionProcessor::openAndRead(const char *filename) // read image file
{
    std::vector<cv::Point> peaks;

#ifdef UNIX
    {
        // Open the file and lock it
        int fd = open(filename, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (fd == -1)
        {
            std::cerr << "Failed to open file" << std::endl;
            return peaks;
        }

        struct flock lock{};
        lock.l_type = F_WRLCK;
        lock.l_whence = SEEK_SET;
        lock.l_start = 0;
        lock.l_len = 0; // Lock the whole file

        if (fcntl(fd, F_SETLK, &lock) == -1)
        {
            std::cerr << "Failed to lock file" << std::endl;
            close(fd);
            return peaks;
        }

        // Do peak finding and create set of measurements - then immediately unlock the file
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        // Do gaussian blur
        if (image.empty())
        {
            std::cerr << "Failed to read image" << std::endl;
            lock.l_type = F_UNLCK;
            fcntl(fd, F_SETLK, &lock);
            close(fd);
            return peaks;
        }
        // cv::waitKey(0);
        cv::GaussianBlur(image, image, cv::Size(3, 3), 0);
        cv::imshow("Image", image);
        peaks = findPeaks(image);
        // Unlock the file
        lock.l_type = F_UNLCK;
        if (fcntl(fd, F_SETLK, &lock) == -1)
        {
            std::cerr << "Failed to unlock file" << std::endl;
            close(fd);
            return peaks;
        }

        close(fd);
    }
#elif WINDOWS
    {
        // Open the file and lock it
        HANDLE hFile = CreateFile(
            filename,                     // File name
            GENERIC_READ | GENERIC_WRITE, // Desired access
            0,                            // Share mode
            NULL,                         // Security attributes
            OPEN_ALWAYS,                  // Creation disposition
            FILE_ATTRIBUTE_NORMAL,        // Flags and attributes
            NULL                          // Template file handle
        );

        if (hFile == INVALID_HANDLE_VALUE)
        {
            std::cerr << "Failed to open file" << std::endl;
            return peaks;
        }

        OVERLAPPED overlapped = {0};
        if (!LockFileEx(
                hFile,
                LOCKFILE_EXCLUSIVE_LOCK,
                0,
                MAXDWORD,
                MAXDWORD,
                &overlapped))
        {
            std::cerr << "Failed to lock file" << std::endl;
            CloseHandle(hFile);
            return peaks;
        }
        // Do peak finding and create set of measurements - then immediately unlock the file
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            std::cerr << "Failed to read image" << std::endl;
            UnlockFileEx(hFile, 0, MAXDWORD, MAXDWORD, &overlapped);
            CloseHandle(hFile);
            return peaks;
        }
        peaks = findPeaks(image);
        // Unlock the file
        if (!UnlockFileEx(
                hFile,
                0,
                MAXDWORD,
                MAXDWORD,
                &overlapped))
        {
            std::cerr << "Failed to unlock file" << std::endl;
            CloseHandle(hFile);
            return peaks;
        }

        CloseHandle(hFile);
    }
#endif
    return peaks;
}

void VisionProcessor::readAndDisplayRawImg(const char *filename) // read img file and display it
{
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cerr << "Failed to read image" << std::endl;
        return;
    }
    std::string imgData = "";
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            uchar intensity = image.at<uchar>(i, j);
            imgData += std::to_string(intensity) + ",";
        }
        imgData += "\n";
    }
    exportValsAsCSV("../testingscripts/generateimages/imgoutput/imageData.csv", imgData);
    cv::imshow("Image", image);
    cv::waitKey(0);
}

int VisionProcessor::exportValsAsCSV(const std::string &filename, const std::string &pixelVals)
{
    std::ofstream file;
    try
    {
        file.open(filename);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file" << std::endl;
            return -1;
        }

        // Convert pixel values to CSV format
        std::ostringstream oss;
        for (size_t i = 0; i < pixelVals.size(); ++i)
        {
            oss << pixelVals[i];
            // if (i != pixelVals.size() - 1)
            // {
            //     oss << ",";
            // }
        }

        file << oss.str();
        file.close();
    }
    catch (const std::ofstream::failure &e)
    {
        std::cerr << "Exception writing to file: " << e.what() << std::endl;
        if (file.is_open())
        {
            file.close();
        }
        return -1;
    }

    return 0;
}

std::vector<std::pair<int, cv::Point>> VisionProcessor::readTruthFile(const char *filename)
{
    std::vector<std::pair<int, cv::Point>> truthVals;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return truthVals;
    }
    json jsonData;
    file >> jsonData;
    file.close();
    try
    {
        for (const auto &item : jsonData)
        {
            int id = item["id"];
            int x = item["x"];
            int y = item["y"];
            // Optionally, you can also read the "time" field if needed
            // int time = item["time"];

            truthVals.push_back(std::make_pair(id, cv::Point(x, y)));
        }
    }
    catch (const json::exception &e)
    {
        std::cerr << "Failed to read JSON file. is it in a valid format? Error: " << e.what() << std::endl;
        return truthVals;
    }
    return truthVals;
}

/**
 * @brief Open the image file and load the data, locking the file from other processes.
 *
 * This function opens the file and loads the data, locking the file from other processes.
 * If the file is being written to by another process, the function will wait until the file is available.
 * This makes use of the fcntl system call to lock the file for unix, and the LockFileEx function for windows.
 *
 * @param filename The name of the file to open and read.
 * @return the set of peaks found in the image.
 */
std::pair<std::vector<cv::Point>, std::vector<std::pair<int, cv::Point>>> VisionProcessor::getPeaksWithTruth(const char *filename, const char *truthFileJson)
{
    // read in the image and get peaks from it
    std::vector<cv::Point> peaks = openAndRead(filename);
    // read in the truth file and get the truth values
    std::vector<std::pair<int, cv::Point>> truthVals = readTruthFile(truthFileJson);
    return std::make_pair(peaks, truthVals);
}

// int main(int argc, char *argv[])
// {
//     std::cout << "Running test of peak finding" << std::endl;
//     if (argc < 2)
//     {
//         std::cerr << "Usage: " << argv[0] << " <image_file>" << std::endl;
//         return -1;
//     }
//     std::cout << "Running test of peak finding" << std::endl;
//     readAndDisplayRawImg(argv[1]);
//     std::vector<statistic> peaks = openAndRead(argv[1]);
//     std::cout << "Found " << peaks.size() << " peaks" << std::endl;
//     for (const auto &stat : peaks)
//     {
//         std::cout << "Peak at (" << stat.peak.x << ", " << stat.peak.y << ") with value of " << stat.value << std::endl;
//     }
//     return 0;
// }

// assume some track
// create some way to simulate the mWidar reciever
// how can we do peak finding first after we get the image data?

/*
Defining the peaks2 function in matlab:



*/