#include "../include/DataProcessor.h"

using namespace cv;
// using namespace Eigen;

DataProcessor::DataProcessor(int imageSize, std::string filter = "kalman-filter")
{
    IMAGE_SIZE = imageSize;
    initializeStateMatrices();
}
DataProcessor::DataProcessor()
{
    IMAGE_SIZE = 128;
    initializeStateMatrices();
}
DataProcessor::~DataProcessor()
{
    std::cout << "DataProcessor destroyed" << std::endl;
}
std::vector<Object> DataProcessor::getObjects()
{
    return objects;
}

void DataProcessor::printMatrices()
{
    std::cout << "A: " << Eigen::MatrixXd(A) << std::endl;
    std::cout << "Gamma: " << Eigen::MatrixXd(Gamma) << std::endl;
    std::cout << "H: " << Eigen::MatrixXd(H) << std::endl;
    std::cout << "W: " << Eigen::MatrixXd(W) << std::endl;
    std::cout << "R: " << Eigen::MatrixXd(R) << std::endl;
    std::cout << "F: " << Eigen::MatrixXd(F) << std::endl;
    std::cout << "Z: " << Eigen::MatrixXd(Z) << std::endl;
    std::cout << "eZ: " << Eigen::MatrixXd(eZ) << std::endl;
    std::cout << "Q: " << Eigen::MatrixXd(Q) << std::endl;
}

void DataProcessor::initializeStateMatrices()
{
    double scale = 0.03125; // 1/32
    // Initialize the sparse matrices with the appropriate sizes
    A.resize(4, 4);
    Gamma.resize(4, 2);
    H.resize(2, 4);
    W.resize(2, 2);
    R.resize(2, 2);
    F.resize(A.rows(), A.rows());
    Z.resize(A.rows() * 2, A.rows() * 2);
    eZ.resize(A.rows() * 2, A.rows() * 2);
    Q.resize(Gamma.cols(), Gamma.cols());

    // Populate with data
    double speedOfLight = 299792458; // m/s
    double dt = 0.5 * speedOfLight * 667e-12;

    // Insert elements into A
    A.insert(0, 0) = 1.0;
    A.insert(0, 1) = dt;
    A.insert(2, 2) = 1.0;
    A.insert(2, 3) = dt;

    // Insert elements into Gamma
    Gamma.insert(1, 0) = 1.0;
    Gamma.insert(3, 1) = 1.0;

    // Insert elements into H
    H.insert(0, 0) = 1.0;
    H.insert(1, 2) = 1.0;

    // Insert elements into W
    W.insert(0, 0) = 1.0;
    W.insert(1, 1) = 1.0;
    W = W * 5; // Adjust as needed

    // Insert elements into R
    R.insert(0, 0) = 1.0;
    R.insert(1, 1) = 1.0;
    // R = R * 1e-4; // Adjust as needed

    // convert meters to pixels
    convertMetersToPx(A, scale);
    convertMetersToPx(Gamma, scale);
    convertMetersToPx(H, scale);
    convertMetersToPx(W, scale);
    convertMetersToPx(R, scale);

    // create Dense matrices so we can compute the matrix exponential
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    Eigen::MatrixXd Gamma_dense = Eigen::MatrixXd(Gamma);
    Eigen::MatrixXd W_dense = Eigen::MatrixXd(W);

    // Create block matrix Z
    Z = createBlockMatrix(A_dense, Gamma_dense, W_dense);
    // Convert sparse matrix Z to dense matrix
    Eigen::MatrixXd denseZ = Eigen::MatrixXd(Z);
    // Compute the matrix exponential
    eZ = manualMatrixExponential(denseZ, 10);
    // Convert the sparse matrix exponential to a dense matrix
    Eigen::MatrixXd eZDense = Eigen::MatrixXd(eZ);
    // Extract F and Q
    Eigen::MatrixXd F_dense = eZDense.topRightCorner(4, 4);
    Eigen::MatrixXd Q_dense = F_dense * eZDense.bottomRightCorner(4, 4) * F_dense.transpose() + eZDense.topLeftCorner(4, 4);
    // Convert F and Q back to sparse matrices
    F = F_dense.sparseView();
    Q = Q_dense.sparseView();
}

Eigen::SparseMatrix<double> DataProcessor::manualMatrixExponential(Eigen::MatrixXd &A, int terms = 10)
{
    Eigen::MatrixXd result = Eigen::MatrixXd::Identity(A.rows(), A.cols()); // Initialize result as I
    Eigen::MatrixXd term = Eigen::MatrixXd::Identity(A.rows(), A.cols());   // First term (A^0 / 0!)
    for (int n = 1; n < terms; ++n)
    {
        term = term * A / n; // Compute A^n / n!
        result += term;      // Add the term to the result
    }
    return result.sparseView();
}

Eigen::SparseMatrix<double> DataProcessor::createBlockMatrix(const Eigen::MatrixXd &A,
                                                             const Eigen::MatrixXd &Gamma,
                                                             const Eigen::MatrixXd &W)
{
    // Determine the sizes of the input matrices
    int rowsA = A.rows();
    int colsA = A.cols();
    int rowsGamma = Gamma.rows();
    int colsGamma = Gamma.cols();
    int rowsW = W.rows();
    int colsW = W.cols();

    // Create a larger matrix to hold the block matrix
    Eigen::MatrixXd blockMatrix(rowsA * 2, colsA * 2);

    // Initialize the block matrix with zeros
    blockMatrix.setZero();

    // Place matrix A in the top-left corner
    blockMatrix.topLeftCorner(rowsA, colsA) = A;

    // Place matrix Gamma in the top-right corner
    blockMatrix.topRightCorner(rowsA, colsGamma) = Gamma;

    // Place matrix W in the bottom-right corner
    blockMatrix.bottomRightCorner(rowsW, colsW) = W;

    // Convert the dense block matrix to a sparse matrix
    Eigen::SparseMatrix<double> sparseBlockMatrix = blockMatrix.sparseView();

    return sparseBlockMatrix;
}

void DataProcessor::processMatFile(mat_t *matfp, matvar_t *matvar, Eigen::SparseMatrix<double> *sparseMat)
{
    if (matvar == nullptr)
    {
        std::cerr << "Variable 'A' not found in MAT file." << std::endl;
        Mat_Close(matfp);
        return;
    }

    if (matvar->rank != 2 || matvar->data_type != MAT_T_DOUBLE)
    {
        std::cerr << "Unsupported matrix format." << std::endl;
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return;
    }

    size_t rows = matvar->dims[0];
    size_t cols = matvar->dims[1];

    // Check if the matrix is sparse
    if (matvar->isComplex)
    {
        std::cerr << "Complex matrices are not supported." << std::endl;
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return;
    }

    if (matvar->class_type == MAT_C_SPARSE)
    {
        mat_sparse_t *sparse = static_cast<mat_sparse_t *>(matvar->data);
        double *values = static_cast<double *>(sparse->data);
        unsigned int *ir = sparse->ir; // Row indices
        unsigned int *jc = sparse->jc; // Column pointers

        size_t nzmax = sparse->nzmax; // Number of non-zero elements

        // Create an Eigen sparse matrix
        sparseMat->resize(rows, cols);

        // Use triplets to populate the Eigen sparse matrix
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nzmax);

        for (size_t col = 0; col < cols; ++col)
        {
            for (int k = jc[col]; k < jc[col + 1]; ++k)
            {
                int row = ir[k];
                double value = values[k];

                if (!std::isnan(value))
                {
                    triplets.emplace_back(row, col, value);
                }
            }
        }

        // Set the values in the Eigen sparse matrix
        sparseMat->setFromTriplets(triplets.begin(), triplets.end());

        // Now eigenSparseMatrix contains the sparse matrix data
        std::cout << "Eigen sparse matrix successfully created." << std::endl;
    }
    else
    {
        std::cerr << "Matrix is not sparse." << std::endl;
    }

    Mat_VarFree(matvar);
    Mat_Close(matfp);
}

void DataProcessor::convertMatFileToEigenSparseMatrix(const char *fileName, Eigen::SparseMatrix<double> *sparseMat)
{
    mat_t *matfp = Mat_Open(fileName, MAT_ACC_RDONLY);
    if (matfp == nullptr)
    {
        std::cerr << "Error opening MAT file: " << fileName << std::endl;
        return;
    }

    matvar_t *matvar = Mat_VarRead(matfp, "A");
    processMatFile(matfp, matvar, sparseMat);
}

/**
 * @brief Creates and destorys objects based on the peaks found in the image.
 *
 * This function takes the peaks found in the image and creates for each peak if it doesn't already exist an object.
 * If there are more peaks than objects, the function will remove the objects that are too close to the border.
 * Otherwise, it will skip the deletion of the object, as it is assumed that the object is still in the image.
 *
 * @param peaks The peaks found in the image.
 * @return A vector of objects created from the peaks.
 */
std::vector<Object> DataProcessor::createObjectsFromPeaks(std::vector<cv::Point> &peaks)
{
    int borderTolerance = 3;
    if (peaks.empty())
    {
        std::cerr << "No peaks found." << std::endl;
        return this->objects;
    }

    // if there are less peaks than objects, remove objects that are closest to the border (within the tolerance)
    for (int i = peaks.size(); i < this->objects.size(); i++)
    {

        for (auto it = this->objects.begin(); it != this->objects.end();)
        {
            if (it->getPixelPosition().x < borderTolerance || it->getPixelPosition().x > IMAGE_SIZE - borderTolerance ||
                it->getPixelPosition().y < borderTolerance || it->getPixelPosition().y > IMAGE_SIZE - borderTolerance)
            {
                it = this->objects.erase(it);
                break;
            }
            else
            {
                ++it;
            }
        }
    }

    // Create new objects if there are more peaks than objects
    while (peaks.size() > this->objects.size())
    {
        Eigen::VectorXd x(4);
        x << peaks.at(this->objects.size()).x, 0, peaks.at(this->objects.size()).y, 0;
        Eigen::MatrixXd P = Eigen::MatrixXd::Identity(4, 4);
        this->objects.push_back(Object(x, P, this->objects.size()));
    }

    return this->objects;
}

/**
 * @brief Maps the set of objects to the measurements found in the image, and updates the objects' state.
 *
 * This function takes the set of objects and the peaks found in the image and maps the objects to the nearest peak.
 * It updates the objects' state based on the new peak positions, and returns a vector of pairs containing the peak and the object.
 *
 * Note: The map has a copy of the object, so the object's state is not updated in place.
 *
 * @param objects The set of objects to be updated.
 * @param measurements The peaks found in the image.
 * @return A vector of pairs containing the peak and the object.
 */
std::vector<std::pair<cv::Point, Object>> DataProcessor::nearestNeighborMapping(std::vector<cv::Point> &measurements)
{
    std::vector<std::pair<cv::Point, Object>> peakToObject;
    for (const auto &peak : measurements)
    {
        Object *nearestObj = nullptr;
        double minDist = std::numeric_limits<double>::max();
        for (auto &obj : this->objects)
        {
            double dist = cv::norm(peak - obj.getPixelPosition());
            if (dist < minDist)
            {
                minDist = dist;
                nearestObj = &obj;
            }
        }
        if (nearestObj != nullptr)
        {
            peakToObject.push_back(std::make_pair(peak, *nearestObj));
            // nearestObj->setMeasurementPixelPos(peak, 0.1); // should set the velocity based on the last position
        }
    }
    // give the right answer, do the filter update and toss the rest
    // pass in a list of {object id : peak}
    // !!!!!!!! -- just to check if filter is working correctly first
    return peakToObject;
}

std::pair<cv::Point, std::pair<int, cv::Point>> DataProcessor::findClosestPointToTruth(std::vector<cv::Point> &measurements, std::pair<int, cv::Point> &truthPoint)
{
    cv::Point closestPoint;
    double minDist = std::numeric_limits<double>::max();
    for (const auto &peak : measurements)
    {
        double dist = cv::norm(peak - truthPoint.second);
        if (dist < minDist)
        {
            minDist = dist;
            closestPoint = peak;
        }
    }
    return std::make_pair(closestPoint, truthPoint);
}

/**
 * @brief Takes a set of peaks found in the image, and creates objects based on the truth data peaks
 *
 * This function takes all of the peaks found in the image, and based on the truth data peaks, creates a
 * set of objects that are associated with the truth data peaks. The function returns a vector of pairs,
 * with the Object containing the state of the object and its estimated position (given by the image)
 * and the cv::Point containing the ground truth position of the object.
 *
 * @param measurements The set of peaks found in the image
 * @param truthData The set of truth data peaks
 * @return A vector of pairs containing the estimated position of the object and the ground truth position
 */
std::vector<std::pair<cv::Point, Object *>> DataProcessor::truthDataMapping(std::vector<cv::Point> &measurements, std::vector<std::pair<int, cv::Point>> &truthData)
{
    // find the closest peak to the truth data
    std::vector<std::pair<cv::Point, std::pair<int, cv::Point>>> closestPoints; // (peak, (truthID, truthPeak))
    std::vector<cv::Point> measurementPoints = measurements;
    closestPoints.reserve(truthData.size());
    for (auto &truth : truthData)
    {
        closestPoints.push_back(findClosestPointToTruth(measurementPoints, truth));
    }
    // create an object for each truth data point if it doesn't already exist
    std::vector<std::pair<cv::Point, Object *>> objectMapping;
    for (const auto &closest : closestPoints)
    {
        bool exists = false;
        for (auto &obj : this->objects)
        {
            if (obj.getID() == closest.second.first)
            {
                exists = true;
                objectMapping.emplace_back(closest.first, &obj);
                Eigen::VectorXd z(2);
                z << closest.first.x, closest.first.y;
                obj.appendMeasurementStateVector(z);
                break;
            }
        }
        if (!exists)
        {
            Eigen::VectorXd x(4);
            x << closest.first.x, 0, closest.first.y, 0; // object gets the measurement, not the truth data
            Eigen::MatrixXd P = Eigen::MatrixXd::Identity(4, 4);
            this->objects.emplace_back(x, P, this->objects.size());
        }
    }
    return objectMapping;
}

template <typename eig_t>
void DataProcessor::convertMetersToPx(eig_t &eigen_type, double scale)
{
    eigen_type *= scale;
}

/**
 * @brief Propogates the state of objects based on the new peak positions.
 *
 * This function is run upon the completion of the vision processor's peak detection. It takes the new peak positions
 * and updates the state estimation of the objects based on the filter type specified. The filter type can be one of the following:
 * - Markov Chain (uses only x and y position)
 * - Kalman Filter (uses x, y, x velocity, y velocity)
 * - Particle Filter (uses x, y, x velocity, y velocity)
 *
 * @param objects The set of objects to be updated.
 * @param filter The type of filter to be used for state estimation. Default is a kalman filter.
 */
void DataProcessor::propogateState(Object &obj, Eigen::VectorXd &measurement, std::string filter = "kalman-filter")
{
    obj.setMeasurementState(measurement);

    if (filter == "markov-chain")
    {
        markovUpdate(obj, measurement); // use x, y
    }
    else if (filter == "kalman-filter")
    {
        kalmanUpdate(obj, measurement); // use x, xdot, y, ydot
    }
    else if (filter == "particle-filter")
    {
        particleUpdate(obj, measurement); // use x, xdot, y, ydot
    }
    else
    {
        std::cerr << "Invalid filter type." << std::endl;
    }
}

void DataProcessor::markovUpdate(Object &obj, Eigen::VectorXd &measurement)
{
    // TODO: Implement markov chain update step
}

/**
 * @brief Performs a kalman filter update step on the object, updating its state vector and covariance matrix.
 *
 * This function takes an object and a measurement and performs a kalman filter update step on the object.
 * It is designed to be called multiple times for each object in the set of objects.
 * The function updates the state vector and covariance matrix of the object based on the measurement.
 *
 *
 * @param obj The object to be updated
 * @param measurement The measurement vector to be used for the update
 */
void DataProcessor::kalmanUpdate(Object &obj, Eigen::VectorXd &measurement)
{
    Eigen::VectorXd x_minus = F * obj.getStateVector();
    Eigen::MatrixXd P_minus = F * obj.getStateCovariance() * F.transpose() + Q;
    Eigen::MatrixXd inov_cov = H * obj.getStateCovariance() * H.transpose() + R;
    Eigen::MatrixXd K = obj.getStateCovariance() * H.transpose() * inov_cov.inverse();
    Eigen::VectorXd inov = measurement - H * x_minus;

    obj.setStateVector(x_minus + K * inov);
    obj.setStateCovariance((Eigen::MatrixXd::Identity(4, 4) - K * H) * P_minus);
}

void DataProcessor::particleUpdate(Object &obj, Eigen::VectorXd &measurement)
{
    // TODO: Implement particle filter update step
}

void DataProcessor::prettyPrintObjectData()
{
    for (auto &obj : objects)
    {
        std::cout << "Object " << obj.getID() << "\nx:\n"
                  << obj.getStateVector() << "\nz:\n"
                  << obj.getMeasurementState() << std::endl;
    }
}
