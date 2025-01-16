/**
 * mWidar Simulator implementation -- Simulator.cpp
 * Compile and run seperately from the main program
 *
 * To compile:
 * /usr/bin/clang++ -std=gnu++14 -fcolor-diagnostics -fansi-escape-codes
 *          -g mWidar/simulator/Simulator.cpp
 *          -o mWidar/simulator/Simulator
 *          -I /usr/local/include/eigen3/
 * To run:
 * ./Simulator -args
 *
 * @param objects A vector of object data, where each object is defined by a list:
 *        [x0, y0, vx0, vy0, ax0, ay0], representing the initial position (x0, y0),
 *        velocity (vx0, vy0), and acceleration (ax0, ay0) in a 2D plane. Each object
 *        is stored as a structure or class with appropriate data fields.
 *
 * @param time_step A float representing the time step for the simulation. The default
 *        value is 0.2.
 *
 * @param output_file A string representing the file path to save the generated image.
 *        The default output path is "output/img".
 *
 * @param display_image A boolean flag that, when set to true, will display the simulation
 *        image after it is generated. The default is false (do not display).
 *
 * @param export_truth A boolean flag that, when set to true, will export the object
 *        coordinates (rounded to the nearest integer) to a file. The default is false.
 *
 * @param use_shared_memory A boolean flag that, when set to true, will use shared memory
 *        for storing image data, which can help optimize performance when working with
 *        large datasets. The default is false (do not use shared memory).
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <vector>

#include "opencv2/opencv.hpp"
#include "Eigen/Dense"

 // Global Variables 
const int IMAGE_SIZE = 128;
Eigen::MatrixXd M, G;

/**
  * Object class to store object data
  *
  * @param x A double representing the x-coordinate of the object
  * @param y A double representing the y-coordinate of the object
  * @param vx A double representing the x-velocity of the object
  * @param vy A double representing the y-velocity of the object
  * @param ax A double representing the x-acceleration of the object
  * @param ay A double representing the y-acceleration of the object
  *
  * @return None
  *
  * @example
  * Object obj = Object(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
  * obj.update(0.2)
  * pos = obj.get_position()
  * vel = obj.get_velocity()
  * acc = obj.get_acceleration()
  * obj.set_position(Eigen::Vector2f(1.0, 1.0))
  * obj.set_velocity(Eigen::Vector2f(0.0, 0.0))
  * obj.set_acceleration(Eigen::Vector2f(1.0, 1.0))
  */
class Object {
public:
    double x, y, vx, vy, ax, ay;
    std::string ID;
    Object(double x0, double y0, double vx0, double vy0, double ax0, double ay0, std::string name) {
        x = x0;
        y = y0;
        vx = vx0;
        vy = vy0;
        ax = ax0;
        ay = ay0;
        ID = name;
    }

    void update(float time_step) {
        x += vx * time_step + 0.5 * ax * time_step * time_step;
        y += vy * time_step + 0.5 * ay * time_step * time_step;
        vx += ax * time_step;
        vy += ay * time_step;
    }

    // Getter Functions -- return pairs of values in Eigen::Vector2f
    Eigen::Vector2f get_position() {
        return Eigen::Vector2f(x, y);
    }

    Eigen::Vector2f get_velocity() {
        return Eigen::Vector2f(vx, vy);
    }

    Eigen::Vector2f get_acceleration() {
        return Eigen::Vector2f(ax, ay);
    }

    std::string get_ID() {
        return ID;
    }

    void set_position(Eigen::Vector2f pos) {
        x = pos(0);
        y = pos(1);
    }

    void set_velocity(Eigen::Vector2f vel) {
        vx = vel(0);
        vy = vel(1);
    }

    void set_acceleration(Eigen::Vector2f acc) {
        ax = acc(0);
        ay = acc(1);
    }

};


/**
 * @brief Display an image using OpenCV
 *
 * This function takes an Eigen::MatrixXf image and displays it using OpenCV.
 * The image is first normalized to the range [0, 255] and then converted to
 * an 8-bit unsigned integer format for display.
 *
 * @param image An Eigen::MatrixXf representing the image data
 *
 * @return None
 */
void displayimage(const Eigen::MatrixXf& image) {
    /**
     * @brief Display an image using OpenCV
     *
     * @param image An Eigen::MatrixXf representing the image data
     *
     * @return None
     */

     //  Convert Eigen matrix FIRST to std::vector, then to OpenCV Mat
    std::vector<float> image_vector(image.data(), image.data() + image.size());
    cv::Mat img(image.rows(), image.cols(), CV_32FC1, image_vector.data());

    // Normalize image for display
    cv::Mat processedImage = img.clone();
    // Crop image from 128x128 to middle 64x64
    // processedImage = processedImage(cv::Rect(32, 32, 64, 64));

    // Normalize the image in-place
    cv::normalize(processedImage, processedImage, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Apply the color map directly to the normalized image
    cv::applyColorMap(processedImage, processedImage, cv::COLORMAP_JET);

    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Image", 800, 800);

    // Display image
    cv::imshow("Image", processedImage);
    // Resize window

    // Wait for 1 second -- NOT KEY PRESS
    cv::waitKey(1);

}

/**
 * @brief Imports a binary file containing matrix data into an Eigen::MatrixXd object.
 *
 * This function reads binary data from a file and populates an Eigen::MatrixXd matrix.
 * The function handles row/column major differences between numpy (row major) and
 * Eigen (column major) by performing an in-place transpose after reading.
 *
 * @param path File path to the binary file containing matrix data
 * @param matrix Reference to the Eigen::MatrixXd that will store the imported data
 * @param row Number of rows in the target matrix
 * @param col Number of columns in the target matrix
 *
 * @throws std::runtime_error If the file cannot be opened or read
 *
 * @note The binary file should contain raw double values in row-major order
 * @note The matrix parameter will be resized to match the specified dimensions
 *
 * Example:
 * @code
 *     Eigen::MatrixXd matrix;
 *     importMatrix("data.bin", matrix, 3, 4);
 * @endcode
 */
void importMatrix(std::string path, Eigen::MatrixXd& matrix, int row, int col) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    // Allocate memory for the matrix -- (row major order)
    matrix.resize(col, row);

    // Read the entire binary file into the matrix
    file.read(reinterpret_cast<char*>(matrix.data()),
        matrix.size() * sizeof(double));

    // Resize matrix (correct dimensions)
    matrix.transposeInPlace();

    if (!file) {
        throw std::runtime_error("Failed to read data from file: " + path);
    }

    file.close();
}



/**
 * @brief Propagates the distribution of objects in the scene through measurement and recovery matrices
 *
 * This function takes an input distribution matrix S and transforms it using measurement matrix M
 * and recovery matrix G to simulate the radar measurement and recovery process.
 * The transformation follows these steps:
 * 1. Flattens the input distribution matrix S into a column vector
 * 2. Applies measurement matrix M to get simulated measurements
 * 3. Applies recovery matrix G to reconstruct the image
 * 4. Reshapes the result back to a 128x128 matrix
 *
 * @param S The input 128x128 matrix representing the object distribution in the scene
 * @param signal The output 128x128 matrix to store the recovered signal/image
 *
 * @note Assumes global measurement matrix M and recovery matrix G are properly initialized
 * @note Input and output matrices must be 128x128 dimensional
 */
void propogateDistribution(const Eigen::MatrixXd& S, Eigen::MatrixXd& signal) {
    // Compute signal using intermediate vector
    Eigen::MatrixXd S_flat = Eigen::Map<Eigen::MatrixXd>((double*)S.transpose().data(), 128 * 128, 1);
    Eigen::MatrixXd signal_flat = M * S_flat;
    signal_flat = G.transpose() * signal_flat;
    signal = Eigen::Map<Eigen::MatrixXd>((double*)signal_flat.data(), 128, 128);
}

// Test the Simulator -- include arguments

int main(int argc, char* argv[]) {
    std::cout << "Starting Simulator" << std::endl;

    // Import matricies
    std::cout << "Importing Matricies" << std::endl;
    importMatrix("../sampling.bin", M, 4096, 16384);
    importMatrix("../recovery.bin", G, 4096, 16384);
    std::cout << "Matricies Imported" << std::endl;

    // Default values
    float time_step = 1.0;
    std::string output_file = "output/img";
    bool display_image = true;
    bool export_truth = false;
    bool use_shared_memory = false;

    // Parse arguments
    // TODO: DO THIS LATER
    Eigen::MatrixXd S; // Will be 128x128
    std::vector<Object> objects;

    objects.push_back(Object(66.0, 66.0, 1.0, 0, 0.0, 0.0, "Object1"));
    objects.push_back(Object(63.0, 63.0, -1.0, 0, 0.0, 0.0, "Object2"));
    // objects.push_back(Object(64.0, 34.0, -1.0, 0, 0.0, 0.0, "Object2"));
    // objects.push_back(Object(94.0, 64.0, 0.0, 1.0, 0.0, 0.0, "Object3"));
    // objects.push_back(Object(64.0, 94.0, 0.0, -1.0, 0.0, 0.0, "Object4"));

    int MaxTimestep = 100; // TODO: Dynamically calculate this for last timestep object will be in frame
    for (int i = 0; i < MaxTimestep; i++) {
        S = Eigen::MatrixXd::Zero(128, 128);
        std::cout << "Time Step: " << i << std::endl;
        for (int obj = 0; obj < objects.size(); obj++) {
            std::cout << "Object: " << objects[obj].get_ID();
            std::cout << " at position: " << objects[obj].get_position().transpose();
            std::cout << std::endl;

            if (objects[obj].get_position()(0) < 128 && objects[obj].get_position()(1) < 128 && objects[obj].get_position()(0) >= 0 && objects[obj].get_position()(1) >= 0) {
                int x_pos = (int)objects[obj].get_position()(0);
                int y_pos = (int)objects[obj].get_position()(1);
                S(x_pos, y_pos) = 1;
            }
            objects[obj].update(time_step);
        }

        // Calculate new signal
        Eigen::MatrixXd signal;
        propogateDistribution(S, signal);


        if (display_image) {
            displayimage(signal.cast<float>() * 100.0f);
        }

    }
    std::cout << "Another test" << std::endl;
    return 0;
}


