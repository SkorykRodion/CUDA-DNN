#include "sigmoid_activation.cuh"


float sigmoid(float x) {
    float result;
    if (x >= 45.)
        result = 1.;
    else if (x <= -45.)
        result = 0.;
    else
        result = 1. / (1. + exp(-x));
    return result;
}

void sigmoidActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {
    for (int row = 0; row < Z_y_dim; row++) {
        for (int col = 0; col < Z_x_dim; col++) {
            int index = row * Z_x_dim + col;
            A[index] = sigmoid(Z[index]);
        }
    }
}


__device__ float sigmoidGPU(float x) {
    float result;
    if (x >= 45.f)
        result = 1.f;
    else if (x <= -45.f)
        result = 0.f;
    else
        result = 1.f / (1.f + expf(-x));
    return result;
}

__global__ void sigmoidActivationForwardGPU(float* Z, float* A,
                                         int Z_x_dim, int Z_y_dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        A[index] = sigmoidGPU(Z[index]);
    }
}



SigmoidActivation::SigmoidActivation(std::string name) {
    this->name = name;
    setHasWeights(false);
}

SigmoidActivation::~SigmoidActivation()
{ }

Matrix& SigmoidActivation::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);

    sigmoidActivationForward(Z.data_host.get(), A.data_host.get(), Z.shape.x, Z.shape.y);

    return A;
}

void SigmoidActivation::forwardGPU(Matrix &Z, cudaStream_t stream) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);
    dim3 block_size(1024);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    sigmoidActivationForwardGPU<<<num_of_blocks, block_size, 0, stream>>>(Z.data_device.get(), A.data_device.get(),
                                                                       Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid forward propagation.");
    Z=A;
}

void sigmoidActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {
    for (int row = 0; row < Z_y_dim; row++) {
        for (int col = 0; col < Z_x_dim; col++) {
            int index = row * Z_x_dim + col;
            dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
        }
    }
}


__global__ void sigmoidActivationBackpropGPU(float* Z, float* dA, float* dZ,
                                          int Z_x_dim, int Z_y_dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        dZ[index] = dA[index] * sigmoidGPU(Z[index]) * (1 - sigmoidGPU(Z[index]));
    }
}

Matrix& SigmoidActivation::backprop(Matrix& dA, float learning_rate) {
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    sigmoidActivationBackprop(Z.data_host.get(), dA.data_host.get(), dZ.data_host.get(), Z.shape.x, Z.shape.y);

    return dZ;
}

void SigmoidActivation::update() {
}

NNLayer *SigmoidActivation::copy() const {
    SigmoidActivation* copyLayer = new SigmoidActivation(name);

    copyLayer->A = Matrix(A);
    copyLayer->Z = Matrix(Z);
    copyLayer->dZ = Matrix(dZ);

    return copyLayer;
}

void SigmoidActivation::backpropGPU(Matrix &dA, float learning_rate, cudaStream_t stream) {
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
    sigmoidActivationBackpropGPU<<<num_of_blocks, block_size,0 ,stream>>>(Z.data_device.get(), dA.data_device.get(),
                                                             dZ.data_device.get(),
                                                             Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid back propagation");

    dA = dZ;
}

void SigmoidActivation::updateGPU() {

}

void SigmoidActivation::accumulateDeltaGPU(NNLayer *layer, cudaStream_t stream) {

}
