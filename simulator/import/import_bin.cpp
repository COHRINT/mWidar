/**
 * This is a C++ executable that will read in
 * two *.bin files and save them in an
 * Eigen Matrix
 */

#include "Eigen/Dense"
#include "fstream"
#include "iostream"
#include "string"
#include <stdexcept>

Eigen::MatrixXd testImport(std::string path) {
  /**
   * Take in the path -- return a MatrixXf
   * of the data in the file
   */

  Eigen::MatrixXd import_matrix(3, 3);

  // Open the file
  int row = 0, col = 0;
  FILE* file = fopen(path.c_str(), "rb");
  while (!feof(file)) {
    double value;
    fread(&value, sizeof(double), 1, file);
    import_matrix(row, col) = value;
    col++;
    if (col == 3) {
      col = 0;
      row++;
    }
    if (row == 3) {
      break;
    }
  }
  return import_matrix;
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

int main() {
  std::string line = "----------------------------------------";

  // Test No 1
  std::cout << line << std::endl;
  std::cout << "Reading in the first matrix" << std::endl;

  Eigen::MatrixXd test_matrix(3, 3);
  test_matrix = testImport("../example.bin");
  std::cout << test_matrix << std::endl;

  // Test No 2
  std::cout << line << std::endl;
  std::cout << "Reading the second matrix" << std::endl;
  Eigen::MatrixXd test_matrix2;
  importMatrix("../example.bin", test_matrix2, 3, 3);
  std::cout << test_matrix2 << std::endl;

  // Test No 3
  std::cout << line << std::endl;
  std::cout << "Reading the third matrix" << std::endl;
  Eigen::MatrixXd test_matrix3;
  importMatrix("../example_2x3.bin", test_matrix3, 2, 3);
  std::cout << test_matrix3 << std::endl;

  // Test No 4
  std::cout << line << std::endl;
  std::cout << "Reading the ACTUAL matrix" << std::endl;
  Eigen::MatrixXd sampling, recovery;
  importMatrix("../../sampling.bin", sampling, 4096, 16384);
  std::cout << sampling(1398, 0) << std::endl;
  importMatrix("../../recovery.bin", recovery, 4096, 16384);
  std::cout << recovery(1398, 0) << std::endl;

  std::cout << line << std::endl;
  return 0;
}
