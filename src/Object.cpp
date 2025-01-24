#include "../include/Object.h"
#include <iostream>
Object::Object(int id)
{
    // matrices are set to I, vectors are set to 0
    this->transitionMatrix = Eigen::MatrixXd::Identity(4, 4);
    this->stateVector = Eigen::MatrixXd::Zero(4, 1); // of the form [x, x', y, y']
    this->stateCovariance = Eigen::MatrixXd::Identity(4, 4);
    this->processNoise = Eigen::MatrixXd::Identity(4, 4);
    this->id = id;
    this->measurementState = Eigen::MatrixXd::Zero(4, 1);
    this->innovationVector = Eigen::MatrixXd::Zero(2, 1);
    this->innovationCov = Eigen::MatrixXd::Zero(2, 2);
}

Object::Object(Eigen::VectorXd x, Eigen::MatrixXd P, int id)
{
    this->measurementState = x;
    this->stateVector = x; // upon object creation, the state vector is the same as the measurement state
    this->stateCovariance = P;
    this->id = id;
}
Object::~Object()
{
    // TODO
}
int Object::getID() const
{
    return this->id;
}
Eigen::MatrixXd Object::getStateVector()
{
    return this->stateVector;
}
Eigen::MatrixXd Object::getMeasurementState()
{
    return this->measurementState;
}
Eigen::MatrixXd Object::getInnovationVector()
{
    return this->innovationVector;
}
Eigen::MatrixXd Object::getInnovationCov()
{
    return this->innovationCov;
}
Eigen::MatrixXd Object::getTransitionMatrix()
{
    return this->transitionMatrix;
}
Eigen::MatrixXd Object::getStateCovariance()
{
    return this->stateCovariance;
}
Eigen::MatrixXd Object::getProcessNoise()
{
    return this->processNoise;
}
void Object::setStateVector(Eigen::MatrixXd x)
{
    this->stateVector = x;
}
void Object::setMeasurementState(Eigen::MatrixXd z)
{
    this->measurementState = z;
}
void Object::setInnovationVector(Eigen::MatrixXd y)
{
    this->innovationVector = y;
}
void Object::setInnovationCov(Eigen::MatrixXd py)
{
    this->innovationCov = py;
}
void Object::setTransitionMatrix(Eigen::MatrixXd F)
{
    this->transitionMatrix = F;
}
void Object::setStateCovariance(Eigen::MatrixXd P)
{
    this->stateCovariance = P;
}
void Object::setProcessNoise(Eigen::MatrixXd Q)
{
    this->processNoise = Q;
}
void Object::setMeasurementPixelPos(cv::Point pixelPos, double dt)
{
    // get last position
    int lastX = this->stateVector(0);
    int lastY = this->stateVector(2);
    // calculate the velocity
    int velX = pixelPos.x - lastX / dt;
    int velY = pixelPos.y - lastY / dt;
    this->stateVector(0) = pixelPos.x;
    this->stateVector(2) = pixelPos.y;
    this->stateVector(1) = velX;
    this->stateVector(3) = velY;
}
cv::Point Object::convertStatePosToPixelPos()
{
    cv::Point pixelPos;
    pixelPos.x = this->stateVector(0);
    pixelPos.y = this->stateVector(1);
    return pixelPos;
}

cv::Point Object::convertVectorToPixelPos(Eigen::VectorXd vec)
{
    cv::Point pixelPos;
    pixelPos.x = vec(0);
    pixelPos.y = vec(2);
    return pixelPos;
}

cv::Point Object::getPixelPosition()
{
    cv::Point pixelPos;
    pixelPos.x = int(this->stateVector(0));
    pixelPos.y = int(this->stateVector(2));
    return pixelPos;
}
std::pair<double, double> Object::getPixelVelocity()
{
    std::pair<double, double> pixelVel;
    pixelVel.first = this->stateVector(1);
    pixelVel.second = this->stateVector(3);
    return pixelVel;
}
void Object::setID(int id)
{
    this->id = id;
}
std::vector<Eigen::MatrixXd> Object::getMeasurementStateVector()
{
    return this->measurementStateVector;
}
void Object::appendMeasurementStateVector(Eigen::MatrixXd z)
{
    this->measurementStateVector.push_back(z);
}
bool Object::operator<(const Object &other) const
{
    return id < other.id; // Example comparison based on id
}