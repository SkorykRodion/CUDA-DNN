
#include "matrix.cuh"


Matrix::Matrix(size_t x_dim, size_t y_dim) :
        shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
        device_allocated(false), host_allocated(false)
{ }

Matrix::Matrix(Shape shape) :
        Matrix(shape.x, shape.y)
{ }

void Matrix::allocateCudaMemory() {
    if (!device_allocated) {
        float* device_memory = nullptr;
        cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
        NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
        data_device = std::shared_ptr<float>(device_memory,
                                             [&](float* ptr){ cudaFree(ptr); });
        device_allocated = true;
    }
}

void Matrix::allocateHostMemory() {
    if (!host_allocated) {
        data_host = std::shared_ptr<float>(new float[shape.x * shape.y],
                                           [&](float* ptr){ delete[] ptr; });
        host_allocated = true;
    }
}

void Matrix::allocateMemory() {
    allocateCudaMemory();
    allocateHostMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
    if (!device_allocated && !host_allocated) {
        this->shape = shape;
        allocateMemory();
    }
}

void Matrix::allocateMemoryIfNotAllocatedZero(Shape shape) {
    if (!device_allocated && !host_allocated) {
        this->shape = shape;
        allocateMemory();
        setZeroHost();
        setZeroDevice();
    }
}

void Matrix::copyHostToDevice() {
    if (device_allocated && host_allocated) {
        cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
        NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
    }
    else {
        throw NNException("Cannot copy host data to not allocated memory on device.");
    }
}

void Matrix::copyDeviceToHost() {
    if (device_allocated && host_allocated) {
        cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
        NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
    }
    else {
        throw NNException("Cannot copy device data to not allocated memory on host.");
    }
}

float& Matrix::operator[](const int index) {
    return data_host.get()[index];
}

const float& Matrix::operator[](const int index) const {
    return data_host.get()[index];
}

void Matrix::print() const {
    std::cout << "Host data:" << std::endl;
    for (size_t row = 0; row < shape.y; ++row) {
        for (size_t col = 0; col < shape.x; ++col) {
            std::cout << std::setprecision(4) << data_host.get()[row * shape.x + col] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Device data:" << std::endl;
    float* host_data = new float[shape.x * shape.y];
    cudaMemcpy(host_data, data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t row = 0; row < shape.y; ++row) {
        for (size_t col = 0; col < shape.x; ++col) {
            std::cout << std::setprecision(4) << host_data[row * shape.x + col] << " ";
        }
        std::cout << std::endl;
    }
    delete[] host_data;
}

void Matrix::setZeroHost() {
    if (!host_allocated) {
        allocateHostMemory();
    }
    std::fill(data_host.get(), data_host.get() + shape.x*shape.y, 0.0f);
}

void Matrix::setZeroDevice() {
    if (!device_allocated) {
        allocateCudaMemory();
    }
    cudaMemset(data_device.get(), 0, shape.x*shape.y * sizeof(float));
}


Matrix::Matrix(const Matrix& other) : shape(other.shape) {
    device_allocated = false;
    if (other.device_allocated) {
        allocateCudaMemory();
        cudaMemcpy(data_device.get(), other.data_device.get(), other.shape.x*other.shape.y * sizeof(float), cudaMemcpyDeviceToDevice);
        device_allocated = true;
    }
    host_allocated = false;
    if (other.host_allocated) {
        allocateHostMemory();
        std::copy(other.data_host.get(), other.data_host.get() + other.shape.x*other.shape.y, data_host.get());
        host_allocated = true;
    }
}

bool Matrix::isDeviceAllocated() const {
    return device_allocated;
}

bool Matrix::isHostAllocated() const {
    return host_allocated;
}
