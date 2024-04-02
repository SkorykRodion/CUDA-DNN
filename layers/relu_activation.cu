#include "relu_activation.cuh"



void reluActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {
    for (int row = 0; row < Z_y_dim; row++) {
        for (int col = 0; col < Z_x_dim; col++) {
            int index = row * Z_x_dim + col;
            A[index] = std::max(Z[index], 0.0f);
        }
    }
}

__global__ void reluActivationForwardGPU(float* Z, float* A,
                                      int Z_x_dim, int Z_y_dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        A[index] = fmaxf(Z[index], 0);
    }
}

void reluActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {
    for (int row = 0; row < Z_y_dim; row++) {
        for (int col = 0; col < Z_x_dim; col++) {
            int index = row * Z_x_dim + col;
            if (Z[index] > 0) {
                dZ[index] = dA[index];
            } else {
                dZ[index] = 0;
            }
        }
    }
}

__global__ void reluActivationBackpropGPU(float* Z, float* dA, float* dZ,
                                       int Z_x_dim, int Z_y_dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        if (Z[index] > 0) {
            dZ[index] = dA[index];
        }
        else {
            dZ[index] = 0;
        }
    }
}

ReLUActivation::ReLUActivation(std::string name) {
    this->name = name;
    setHasWeights(false);
}

ReLUActivation::~ReLUActivation() { }

Matrix& ReLUActivation::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);

    reluActivationForward(Z.data_host.get(), A.data_host.get(), Z.shape.x, Z.shape.y);

    return A;
}

void ReLUActivation::forwardGPU(Matrix &Z, cudaStream_t stream) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);
    dim3 block_size(1024);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    reluActivationForwardGPU<<<num_of_blocks, block_size, 0, stream>>>(Z.data_device.get(), A.data_device.get(),
                                                         Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU forward propagation.");
    Z=A;
}

void ReLUActivation::backpropGPU(Matrix &dA, float learning_rate, cudaStream_t stream) {
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(1024);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
    reluActivationBackpropGPU<<<num_of_blocks, block_size, 0, stream>>>(
                            Z.data_device.get(), dA.data_device.get(),
                            dZ.data_device.get(),
                            Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU back propagation");

    dA = dZ;
}

Matrix& ReLUActivation::backprop(Matrix& dA, float learning_rate) {
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    reluActivationBackprop(Z.data_host.get(), dA.data_host.get(),
                           dZ.data_host.get(), Z.shape.x,
                           Z.shape.y);

    return dZ;
}



void ReLUActivation::update() {
}

NNLayer *ReLUActivation::copy() const {

    ReLUActivation* copyLayer = new ReLUActivation(name);

    copyLayer->A = Matrix(A);
    copyLayer->Z = Matrix(Z);
    copyLayer->dZ = Matrix(dZ);

    return copyLayer;
}

void ReLUActivation::updateGPU() {

}

void ReLUActivation::accumulateDeltaGPU(NNLayer *layer, cudaStream_t stream) {

}

