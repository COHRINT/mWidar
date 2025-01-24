#ifndef OBJECT_H
#define OBJECT_H

#include "dataStructures.h"
class Object : public DataStructures
{
public:
    Object(int id); // matrices are set to I, vectors are set to 0
    Object(Eigen::MatrixXd F, Eigen::VectorXd x, Eigen::MatrixXd P, Eigen::MatrixXd Q, int id);
    Object(Eigen::VectorXd x, Eigen::MatrixXd P, int id);
    ~Object();

    // getters
    int getID() const;
    cv::Point getPixelPosition();
    std::pair<double, double> getPixelVelocity();
    Eigen::MatrixXd getStateVector();
    Eigen::MatrixXd getMeasurementState();                    // return the measurement state z (no KF, or HMM)
    std::vector<Eigen::MatrixXd> getMeasurementStateVector(); // return a list of previous measurement states
    Eigen::MatrixXd getInnovationVector();
    Eigen::MatrixXd getInnovationCov();
    Eigen::MatrixXd getTransitionMatrix();
    Eigen::MatrixXd getStateCovariance();
    Eigen::MatrixXd getProcessNoise();

    // setters
    void setVelocity(vector v);
    void setAcceleration(vector a);
    void setStateVector(Eigen::MatrixXd x);
    void setMeasurementState(Eigen::MatrixXd z);
    void setInnovationVector(Eigen::MatrixXd y);
    void setInnovationCov(Eigen::MatrixXd py);
    void setTransitionMatrix(Eigen::MatrixXd F);
    void setStateCovariance(Eigen::MatrixXd P);
    void setProcessNoise(Eigen::MatrixXd Q);
    void setMeasurementPixelPos(cv::Point p, double dt);
    void appendMeasurementStateVector(Eigen::MatrixXd z);
    void removeFromMeasurementStateVector(int index);

    // other methods
    cv::Point convertStatePosToPixelPos();
    vector convertStateVelToPixelVel();
    vector convertStateAccToPixelAcc();
    cv::Point convertVectorToPixelPos(Eigen::VectorXd vec);

    // logical operations
    bool operator<(const Object &other) const;
    bool operator==(const Object &other) const;
    bool operator>(const Object &other) const;

private:
    Eigen::MatrixXd transitionMatrix; // Transition matrix F
    Eigen::VectorXd stateVector;      // State vector x
    Eigen::VectorXd measurementState; // Measurement state z
    Eigen::MatrixXd stateCovariance;  // State covariance P
    Eigen::MatrixXd processNoise;     // Process noise Q
    Eigen::VectorXd innovationVector; // Innovation vector y
    Eigen::MatrixXd innovationCov;    // Innovation covariance py
    std::vector<Eigen::MatrixXd> measurementStateVector;
    int id;
    void setID(int id);
};
#endif // OBJECT_H