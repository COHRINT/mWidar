/*
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

#include "Eigen/Dense"
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


 // Create Object class
class Object {
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

// OpenCV Image Display 

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
    cv::waitKey(100);

}


// Helper function to parse float directly from char buffer
float parse_float(const char*& p) {
    while (*p == ' ' || *p == '\t') ++p;
    char* end;
    float result = strtof(p, &end);
    p = end;
    return result;
}

Eigen::MatrixXf import_matrix(const std::string& file_path) {
    constexpr size_t ROWS = 4096;
    constexpr size_t COLS = 16384;

    // Open file
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Cannot open file");

    struct stat sb;
    if (fstat(fd, &sb) == -1) throw std::runtime_error("Cannot get file size");

    const char* data = static_cast<const char*>(
        mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) throw std::runtime_error("Cannot map file");

    // Pre-allocate matrix with known size
    Eigen::MatrixXf objects_matrix(ROWS, COLS);

    const char* p = data;
    for (size_t row = 0; row < ROWS; ++row) {
        for (size_t col = 0; col < COLS; ++col) {
            while (*p == ' ' || *p == '\t') ++p;
            objects_matrix(row, col) = strtof(p, const_cast<char**>(&p));
        }
        while (*p != '\n' && *p != '\0') ++p;
        ++p;

        if (row % 100 == 0) {
            int progress = (row * 100) / ROWS;
            // std::cout << "\rProgress: " << progress << "%" << std::flush;
        }
    }

    munmap(const_cast<char*>(data), sb.st_size);
    close(fd);

    std::cout << "Matrix Imported" << std::endl;
    return objects_matrix;
}


Eigen::MatrixXf sampleDistribution(Eigen::VectorXf truth, const Eigen::MatrixXf M, const Eigen::MatrixXf G){
    /**
     * @brief Based on the truth matrix, hypothesis the mWidar signal and return it 
     * 
     * @param truth The vector representing the object distribution
     * @param M The matrix representing the mWidar sampling matrix (4096 x 16384)
     * @param G The matrix representing the mWidar recovery matrix (4096 x 16384)
     * 
     * @return The mWidar signal matrix
     */

    // If truth is not 16384x1, return error 
    if (truth.size() != 16384) {
        throw std::runtime_error("Truth vector is not 16384x1");
    }

    // Multiply the truth vector by the mWidar sampling matrix
    Eigen::VectorXf signal = M * truth;
    truth = G.transpose() * signal;

    // Return the mWidar signal matrix
    return Eigen::Map<Eigen::MatrixXf>(truth.data(), 128, 128);
}

// Test the Simulator -- include arguments

int main(int argc, char* argv[]) {
    std::cout << "Starting Simulator" << std::endl;



    // Import matricies
    Eigen::MatrixXf M(4096, 16384);
    Eigen::MatrixXf G(4096, 16384);
    std::cout << "Importing Matricies" << std::endl;
    M = import_matrix("data_txt/recovery-12tx-4096samples-128x128.txt");
    G = import_matrix("data_txt/sampling-12tx-4096samples-128x128.txt");
    std::cout << "Matricies Imported" << std::endl;

    // Matricies are constructed baesd on ROW MAJOR FLATTENING -- so we need to use RowMajorMatrix 
    // put here to avoid confusion with Eigen::MatrixXf
    using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Default values
    float time_step = 1.0;
    std::string output_file = "output/img";
    bool display_image = true;
    bool export_truth = false;
    bool use_shared_memory = false;

    // Parse arguments
    // TODO: DO THIS LATER
    std::vector<Object> objects;

    objects.push_back(Object(34.0, 64.0, 1.0, 0, 0.0, 0.0, "Object1"));
    objects.push_back(Object(64.0, 34.0, -1.0, 0, 0.0, 0.0, "Object2"));
    objects.push_back(Object(94.0, 64.0, 0.0, 1.0, 0.0, 0.0, "Object3"));
    objects.push_back(Object(64.0, 94.0, 0.0, -1.0, 0.0, 0.0, "Object4"));

    int MaxTimestep = 100; // TODO: Dynamically calculate this for last timestep object will be in frame

    RowMajorMatrix S; // Will be 128x128
    for (int i = 0; i < MaxTimestep; i++) {
        S = Eigen::MatrixXf::Zero(128, 128);
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

        Eigen::VectorXf S_flat = Eigen::Map<Eigen::VectorXf>(S.data(), S.size());

        // As a check -- print out index number of 1s in S_flat
        for (int i = 0; i < S_flat.size(); i++) {
            if (S_flat(i) == 1) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl;

        Eigen::MatrixXf S_new = sampleDistribution(S_flat, M, G);

        if (display_image) {
            displayimage(S * 100);
            displayimage(S_new);
        }

    }
    std::cout << "Another test" << std::endl;
    return 0;
}


