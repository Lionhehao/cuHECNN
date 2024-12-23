#pragma once

#include <troy/troy.h>

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

inline bool isNumber(const std::string &str) {
    if (str.empty()) return false;
    char *end = nullptr;
    std::strtod(str.c_str(), &end);
    return end != str.c_str();
}

inline std::vector<std::vector<double>> readCsvToTestDatas(
    const std::string &filename) {
    std::vector<std::vector<double>> test_datas;
    std::ifstream file(filename);
    std::string line;
    size_t i = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        test_datas.push_back(std::vector<double>());
        while (std::getline(ss, value, ',')) {
            if (isNumber(value)) {
                test_datas[i].push_back(std::stod(value));  // 转换为 double
            }
        }
        i++;
    }

    file.close();
    return test_datas;
}

inline std::vector<std::vector<double>> readCsvToWeights(
    const std::string &filename) {
    std::vector<std::vector<double>> weights;
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);
    std::stringstream ss(line);
    std::string key;

    while (std::getline(ss, key, ',')) {
        weights.push_back(std::vector<double>());
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        for (size_t i = 0; i < weights.size(); i++) {
            std::string value;
            if (std::getline(ss, value, ',')) {
                if (isNumber(value)) {
                    weights[i].push_back(std::stod(value));  // 转换为 double
                }
            }
        }
    }

    file.close();
    return weights;
}

/*
Helper function: Prints the parameters in a SEALContext.
*/
inline void print_parameters(const troy::HeContext &context) {
    auto context_data = context.key_context_data().value();

    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data->parms().scheme()) {
        case troy::SchemeType::BFV:
            scheme_name = "BFV";
            break;
        case troy::SchemeType::CKKS:
            scheme_name = "CKKS";
            break;
        case troy::SchemeType::BGV:
            scheme_name = "BGV";
            break;
        default:
            throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: "
              << context_data->parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data->total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus_device = context_data->parms().coeff_modulus();
    troy::utils::Array<troy::Modulus> coeff_modulus(coeff_modulus_device.size(),
                                                    false);
    coeff_modulus.copy_from_slice(coeff_modulus_device);
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++) {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus[coeff_modulus_size - 1].bit_count();
    std::cout << ") bits" << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data->parms().scheme() == troy::SchemeType::BFV ||
        context_data->parms().scheme() == troy::SchemeType::BGV) {
        std::cout << "|   plain_modulus: "
                  << context_data->parms().plain_modulus_host().value()
                  << std::endl;
    }

    std::cout << "\\" << std::endl;
}

/*
Helper function: Prints a vector of floating-point values.
*/
template <typename T>
inline void print_vector(std::vector<T> vec, std::size_t print_size = 4,
                         int prec = 3) {
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    std::size_t slot_count = vec.size();

    std::cout << std::fixed << std::setprecision(prec);
    std::cout << std::endl;
    if (slot_count <= 2 * print_size) {
        std::cout << "    [";
        for (std::size_t i = 0; i < slot_count; i++) {
            std::cout << " " << vec[i]
                      << ((i != slot_count - 1) ? "," : " ]\n");
        }
    } else {
        vec.resize(std::max(vec.size(), 2 * print_size));
        std::cout << "    [";
        for (std::size_t i = 0; i < print_size; i++) {
            std::cout << " " << vec[i] << ",";
        }
        if (vec.size() > 2 * print_size) {
            std::cout << " ...,";
        }
        for (std::size_t i = slot_count - print_size; i < slot_count; i++) {
            std::cout << " " << vec[i]
                      << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

template <typename T>
void generate_lmask(size_t column, std::vector<T> &mask, size_t m,
                    size_t target_column) {
    for (size_t i = column; i < m * target_column; i += target_column) {
        mask[i] = 1;
    }
}

template <typename T>
void generate_rmask(size_t row, std::vector<T> &mask, size_t m, size_t l,
                    size_t target_column) {
    for (size_t i = row * target_column; i < (row + 1) * target_column; i++) {
        mask[i] = 1;
    }
}

template <typename T>
void generate_mask(std::vector<T> &mask, size_t first, size_t last,
                   size_t column) {
    for (size_t i = first * column; i < last * column; i++) {
        mask[i] = 1;
    }
}

template <typename T>
void generate_filter(std::vector<T> &filter, size_t m, size_t n, size_t l) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            filter[i * m * l + j * l + (j + i) % n] = 1;
        }
    }
}

/*
Helper function: Generate a 2D vector of floating-point values.
*/
template <typename T>
inline std::vector<std::vector<T>> generate_rand_matrix(std::size_t row,
                                                        std::size_t column) {
    std::vector<std::vector<T>> matrix(row, std::vector<T>(column, 0));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0, 10);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

template <typename T>
inline std::vector<std::vector<T>> multiply_matrices(
    const std::vector<std::vector<T>> &A,
    const std::vector<std::vector<T>> &B) {
    // Check if matrix dimensions are valid for multiplication
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument(
            "Matrix dimensions do not allow multiplication.");
    }

    // Matrix A is of size m x n, Matrix B is of size n x p
    std::size_t m = A.size();     // Rows in A
    std::size_t n = A[0].size();  // Columns in A = Rows in B
    std::size_t p = B[0].size();  // Columns in B

    // Initialize the result matrix C of size m x p with zeros
    std::vector<std::vector<T>> C(m, std::vector<T>(p, 0));

    // Perform matrix multiplication
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;  // Return the result matrix
}

/*
Helper function: Prints a 2D vector of floating-point values.
*/
template <typename T>
inline void print_matrix(const std::vector<std::vector<T>> &matrix,
                         std::size_t print_size = 4, int prec = 3) {
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    std::size_t row_count = matrix.size();
    std::size_t col_count = row_count > 0 ? matrix[0].size() : 0;

    std::cout << std::fixed << std::setprecision(prec);
    std::cout << std::endl;

    // Print matrix with row and column size checking
    for (std::size_t i = 0; i < row_count; ++i) {
        if (i < print_size || i >= row_count - print_size) {
            std::cout << "    [";
            if (col_count <= 2 * print_size) {
                // Print all columns if small enough
                for (std::size_t j = 0; j < col_count; ++j) {
                    std::cout << " " << matrix[i][j]
                              << ((j != col_count - 1) ? "," : " ");
                }
            } else {
                // Print first few and last few columns with ellipsis in the
                // middle
                for (std::size_t j = 0; j < print_size; ++j) {
                    std::cout << " " << matrix[i][j] << ",";
                }
                std::cout << " ...,";
                for (std::size_t j = col_count - print_size; j < col_count;
                     ++j) {
                    std::cout << " " << matrix[i][j]
                              << ((j != col_count - 1) ? "," : " ");
                }
            }
            std::cout << "]\n";
        } else if (i == print_size) {
            // Print ellipsis for skipped rows
            std::cout << "    ...\n";
        }
    }

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

template <typename T>
inline void encode_matrix_l(std::vector<std::vector<T>> &matrix,
                            std::size_t target_column) {
    for (auto &row : matrix) {
        row.resize(target_column);
    }
}

template <typename T>
inline void encode_matrix_r(std::vector<std::vector<T>> &matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    std::vector<std::vector<T>> transposed(cols, std::vector<T>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    matrix = transposed;
}

inline void pack_matrix(std::vector<std::vector<double>> &matrix,
                        std::vector<std::complex<double>> &vec) {
    int idx = 0;
    for (const auto &row : matrix) {
        std::copy(row.begin(), row.end(), vec.begin() + idx * row.size());
        idx++;
    }
}

template <typename T>
std::vector<std::vector<T>> convertTo2D(const std::vector<T> &vec, int rows,
                                        int cols) {
    std::vector<std::vector<T>> matrix(rows, std::vector<T>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = vec[i * cols + j];
        }
    }

    return matrix;
}

// Function to extract convolution elements
inline void extractConvolutionElements(const std::vector<double> &A,
                                       std::vector<std::complex<double>> &B,
                                       int A_size, int kernel_size,
                                       int stride) {
    int B_cols = ((A_size - kernel_size) / stride + 1) *
                 ((A_size - kernel_size) / stride + 1);

    int col_idx = 0;

    for (int i = 0; i <= A_size - kernel_size; i += stride) {
        for (int j = 0; j <= A_size - kernel_size; j += stride) {
            int element_idx = 0;
            for (int ki = 0; ki < kernel_size; ++ki) {
                for (int kj = 0; kj < kernel_size; ++kj) {
                    B[element_idx * B_cols + col_idx] =
                        (A[(i + ki) * A_size + (j + kj) + 1] / 255.0 - 0.1307) /
                        0.3081;
                    ++element_idx;
                }
            }
            ++col_idx;
        }
    }
}