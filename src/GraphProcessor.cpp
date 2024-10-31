#include "../include/GraphProcessor.h"
#define B_SIZE 128
using namespace cv;

GraphProcessor::GraphProcessor() : b(B_SIZE * B_SIZE)
{
    b.setZero();
    img = cv::Mat(IMAGE_SIZE, IMAGE_SIZE, CV_64F, const_cast<double *>(scaleB(b, UPSCALE_FACTOR).data()));
    cv::Mat originalImg = img.clone();
    processImage(img);
}

GraphProcessor::~GraphProcessor()
{
    cv::destroyAllWindows();
}

cv::Mat GraphProcessor::processImage(const cv::Mat &image)
{
    cv::Mat processedImage = image.clone();

    // Normalize the image in-place
    cv::normalize(processedImage, processedImage, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Apply the color map directly to the normalized image
    cv::applyColorMap(processedImage, processedImage, cv::COLORMAP_JET);

    return processedImage;
}

Eigen::VectorXd GraphProcessor::scaleB(Eigen::VectorXd &b, int scale)
{
    int originalSize = static_cast<int>(std::sqrt(b.size())); // Assuming b represents a square image
    int newSize = originalSize * scale;
    Eigen::VectorXd scaledB(newSize * newSize);
    for (int i = 0; i < originalSize; i++)
    {
        for (int j = 0; j < originalSize; j++)
        {
            // Get the value from the original vector
            double value = b(i * originalSize + j);

            // Map this value to a block in the scaled vector
            for (int di = 0; di < scale; di++)
            {
                for (int dj = 0; dj < scale; dj++)
                {
                    int newI = i * scale + di;
                    int newJ = j * scale + dj;
                    scaledB(newI * newSize + newJ) = value;
                }
            }
        }
    }

    return scaledB;
}

cv::Mat GraphProcessor::updateMap(Eigen::VectorXd &b)
{
    std::ostringstream errorMsg;
    errorMsg << "The size of the vector b (" << b.size() << ") is not equal to the size of the image (" << IMAGE_SIZE << " * " << IMAGE_SIZE << " = " << IMAGE_SIZE * IMAGE_SIZE << ").";
    assert((b.rows() * b.cols() * UPSCALE_FACTOR * UPSCALE_FACTOR == IMAGE_SIZE * IMAGE_SIZE) && errorMsg.str().c_str());

    // Update the class member variable img with the new data
    img = cv::Mat(IMAGE_SIZE, IMAGE_SIZE, CV_64F, const_cast<double *>(scaleB(b, UPSCALE_FACTOR).data()));
    cv::Mat coloredMatrix = processImage(img);
    return coloredMatrix;
}

void GraphProcessor::displayImage(const cv::Mat &image)
{
    while (true)
    {
        cv::imshow("Colored Matrix", image);
        // char key = cv::waitKey(1);
        // if (key == 'q')
        // {
        //     break;
        // }
    }
}

cv::Mat GraphProcessor::drawCircle(cv::Mat &image, const cv::Point &center, int radius, const cv::Scalar &color)
{
    cv::Point scaledCenter = cv::Point(center.x * UPSCALE_FACTOR, center.y * UPSCALE_FACTOR);
    // cv::Mat img = image.clone();
    cv::circle(image, scaledCenter, radius, color, 1);
    return image;
}

cv::Mat GraphProcessor::drawSquare(cv::Mat &image, const cv::Point &center, int size, const cv::Scalar &color)
{
    cv::Point scaledCenter = cv::Point(center.x * UPSCALE_FACTOR, center.y * UPSCALE_FACTOR);
    // cv::Mat img = image.clone();
    cv::rectangle(image, scaledCenter - cv::Point(size, size), scaledCenter + cv::Point(size, size), color, 1);
    return image;
}

cv::Mat GraphProcessor::drawVelocityVector(cv::Mat &image, const cv::Point &center, const cv::Point &velocity, const cv::Scalar &color)
{
    cv::Point scaledCenter = cv::Point(center.x * UPSCALE_FACTOR, center.y * UPSCALE_FACTOR);
    cv::Point scaledVelocity = cv::Point(velocity.x * UPSCALE_FACTOR, velocity.y * UPSCALE_FACTOR);
    // cv::Mat img = image.clone();
    cv::arrowedLine(image, scaledCenter, scaledCenter + scaledVelocity, color, 1);
    return image;
}

cv::Mat GraphProcessor::writeTruthTargetsToImg(cv::Mat &image, std::vector<std::pair<int, cv::Point>> &targets)
{
    cv::Mat img = image.clone();
    for (auto &target : targets)
    {
        img = drawCircle(img, target.second);
    }
    return img;
}

cv::Mat GraphProcessor::writeEstTargetsToImg(cv::Mat &image, std::vector<Object> &targets, bool drawVelocity)
{
    cv::Mat img = image.clone();
    for (auto &target : targets)
    {
        img = drawSquare(img, target.getPixelPosition());
        if (drawVelocity)
        {
            cv::Point velocity = cv::Point(target.getStateVector()(1), target.getStateVector()(3));
            img = drawVelocityVector(img, target.getPixelPosition(), velocity, cv::Scalar(255, 255, 255));
        }
    }
    return img;
}
cv::Mat GraphProcessor::clearImage()
{
    img = originalImg;
    return img;
}

int GraphProcessor::convertXYToBIndex(int x, int y)
{
    return y * B_SIZE + x;
}

void GraphProcessor::resetB()
{
    b.setZero();
}

void GraphProcessor::mapObjectToB(Eigen::VectorXd &b, Object obj)
{
    b[convertXYToBIndex(obj.getPixelPosition().x, obj.getPixelPosition().y)] = 1;
}