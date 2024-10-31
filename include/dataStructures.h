#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
class DataStructures
{
public:
    struct point
    {
        int x;
        int y;
    };

    struct probabilityPoint
    {
        point p;
        double probability;
    };

    struct vector
    {
        double direction; // measured in RAD
        double magnitude; // measured in pixels/sec or pixels/sec^2
    };

    struct rgb
    {
        int r;
        int g;
        int b;
    };
};
#endif // DATASTRUCTURES_H