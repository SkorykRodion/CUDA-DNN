
#ifndef COURSE_TRY_2_MATRIX_CUH
#define COURSE_TRY_2_MATRIX_CUH


#include "shape.cuh"
#include "nn_exeption.cuh"
#include <memory>
#include <iostream>
#include <iomanip>

class Matrix {
private:
    bool device_allocated;
    bool host_allocated;

    void allocateCudaMemory();
    void allocateHostMemory();

public:
    bool isDeviceAllocated() const;

    bool isHostAllocated() const;

    Shape shape;

    std::shared_ptr<float> data_device;
    std::shared_ptr<float> data_host;

    Matrix(size_t x_dim = 1, size_t y_dim = 1);
    Matrix(Shape shape);
    Matrix(const Matrix& other);
    void allocateMemory();
    void allocateMemoryIfNotAllocated(Shape shape);

    void copyHostToDevice();
    void copyDeviceToHost();

    float& operator[](const int index);
    const float& operator[](const int index) const;

    void Matrix::print() const;

    void setZeroHost();

    void setZeroDevice();

    void allocateMemoryIfNotAllocatedZero(Shape shape);
};


#endif //COURSE_TRY_2_MATRIX_CUH
