#include "linear_layer.cuh"

__global__ void addMatrixDividedByScalar(float* A, float* B, int A_rows, int A_cols, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < A_cols) {
        int index = row * A_cols + col;
        atomicAdd(&A[index], B[index] / float(m));
    }
}

void linearLayerForward(float* W, float* A, float* Z, float* b,
                        int W_x_dim, int W_y_dim,
                        int A_x_dim, int A_y_dim,
                        int Z_x_dim, int Z_y_dim) {
    for (int row = 0; row < Z_y_dim; row++) {
        for (int col = 0; col < Z_x_dim; col++) {
            float Z_value = 0;
            for (int i = 0; i < W_x_dim; i++) {
                Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
            }
            Z[row * Z_x_dim + col] = Z_value + b[row];
        }
    }
}

__global__ void linearLayerForwardGPU( float* W, float* A, float* Z, float* b,
                                    int W_x_dim, int W_y_dim,
                                    int A_x_dim, int A_y_dim) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    float Z_value = 0;

    if (row < Z_y_dim && col < Z_x_dim) {

        for (int i = 0; i < W_x_dim; i++) {
            Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
        }
        Z[row * Z_x_dim + col] = Z_value + b[row];
    }
}


void linearLayerBackprop(float* W, float* dZ, float* dA,
                             int W_x_dim, int W_y_dim,
                             int dZ_x_dim, int dZ_y_dim) {
    // W is treated as transposed
    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = W_x_dim;
    for (int col = 0; col < dA_x_dim; col++) {
        for (int row = 0; row < dA_y_dim; row++) {
            float dA_value = 0.0f;
            for (int i = 0; i < W_y_dim; i++) {
                dA_value +=W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
            }
            dA[row * dA_x_dim + col] = dA_value;
        }
    }
}

__global__ void linearLayerBackpropGPU(float* W, float* dZ, float *dA,
                                    int W_x_dim, int W_y_dim,
                                    int dZ_x_dim, int dZ_y_dim) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // W is treated as transposed
    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = W_x_dim;

    float dA_value = 0.0f;

    if (row < dA_y_dim && col < dA_x_dim) {
        for (int i = 0; i < W_y_dim; i++) {
            dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
        }
        dA[row * dA_x_dim + col] = dA_value;
    }
}



void linearLayerComputeAndStoreDeltaWeights(float* dZ, float* A, float* W,
                                            int dZ_x_dim, int dZ_y_dim,
                                            int A_x_dim, int A_y_dim,
                                            float* deltaW, float learning_rate) {

    // A is treated as transposed
    int W_x_dim = A_y_dim;
    int W_y_dim = dZ_y_dim;

    for (int row = 0; row < W_y_dim; row++) {
        for (int col = 0; col < W_x_dim; col++) {
            float dW_value = 0.0f;
            for (int i = 0; i < dZ_x_dim; i++) {
                dW_value += dZ[row * dZ_x_dim + i] *  A[col * A_x_dim + i];
            }
            deltaW[row * W_x_dim + col] -= learning_rate * (dW_value / A_x_dim);
        }
    }
}

__global__ void linearLayerComputeAndStoreDeltaWeightsGPU(float* dZ, float* A, float* W,
                                           int dZ_x_dim, int dZ_y_dim,
                                           int A_x_dim, int A_y_dim,
                                           float* deltaW, float learning_rate) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // A is treated as transposed
    int W_x_dim = A_y_dim;
    int W_y_dim = dZ_y_dim;

    float dW_value = 0.0f;

    if (row < W_y_dim && col < W_x_dim) {
        for (int i = 0; i < dZ_x_dim; i++) {
            dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
        }
        deltaW[row * W_x_dim + col] -= learning_rate * (dW_value / A_x_dim);
    }
}



void linearLayerComputeAndStoreDeltaBias(float* dZ, float* b,
                           int dZ_x_dim, int dZ_y_dim,
                           int b_x_dim, float* deltaBias,
                           float learning_rate) {
        for (int dZ_y = 0; dZ_y < dZ_y_dim; dZ_y++) {
        float bias_update = 0.0f;
        for (int dZ_x = 0; dZ_x < dZ_x_dim; dZ_x++) {
            bias_update += dZ[dZ_y * dZ_x_dim + dZ_x];
        }
        deltaBias[dZ_y] -= learning_rate * (bias_update / dZ_x_dim);
    }
}

__global__ void linearLayerComputeAndStoreDeltaBiasGPU(  float* dZ, float* b,
                                        int dZ_x_dim, int dZ_y_dim,
                                        int b_x_dim, float* deltaBias,
                                        float learning_rate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dZ_x_dim * dZ_y_dim) {
        int dZ_x = index % dZ_x_dim;
        int dZ_y = index / dZ_x_dim;
        atomicAdd(&deltaBias[dZ_y], - learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
    }
}

LinearLayer::LinearLayer(std::string name, Shape W_shape) :
        W(W_shape), b(W_shape.y, 1)
{
    this->name = name;
    b.allocateMemory();
    W.allocateMemory();
    setHasWeights(true);
    initializeBiasWithZeros();
    initializeWeightsRandomly();
}

LinearLayer::~LinearLayer()
{ }

void LinearLayer::initializeWeightsRandomly() {
    std::default_random_engine generator;
    std::normal_distribution<float> normal_distribution(0.0, 1.0);

    for (int x = 0; x < W.shape.x; x++) {
        for (int y = 0; y < W.shape.y; y++) {
            W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
        }
    }

    W.copyHostToDevice();
}

void LinearLayer::initializeBiasWithZeros() {
    for (int x = 0; x < b.shape.x; x++) {
        b[x] = 0;
    }

    b.copyHostToDevice();
}

Matrix& LinearLayer::forward(Matrix& A) {
    assert(W.shape.x == A.shape.y);

    this->A = A;
    Shape Z_shape(A.shape.x, W.shape.y);
    Z.allocateMemoryIfNotAllocated(Z_shape);

    linearLayerForward(W.data_host.get(), A.data_host.get(),
                       Z.data_host.get(), b.data_host.get(),
                       W.shape.x, W.shape.y,
                       A.shape.x, A.shape.y,
                       Z.shape.x, Z.shape.y);

    return Z;
}

void LinearLayer::forwardGPU(Matrix &A, cudaStream_t stream) {
    assert(W.shape.x == A.shape.y);
    this->A = A;
    Shape Z_shape(A.shape.x, W.shape.y);
    Z.allocateMemoryIfNotAllocated(Z_shape);

    dim3 block_size(16, 16);
    dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x,
                           (Z.shape.y + block_size.y - 1) / block_size.y);
    linearLayerForwardGPU<<<num_of_blocks, block_size, 0, stream>>>( W.data_device.get(),
                                                       A.data_device.get(),
                                                       Z.data_device.get(),
                                                       b.data_device.get(),
                                                       W.shape.x, W.shape.y,
                                                       A.shape.x, A.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation.");
    A=Z;
}


void LinearLayer::backpropGPU(Matrix &dZ, float learning_rate, cudaStream_t stream) {
    dA.allocateMemoryIfNotAllocated(A.shape);
    computeAndStoreBackpropErrorGPU(dZ, stream);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform back propagation.");


    deltaBias.allocateMemoryIfNotAllocatedZero(W.shape);
    ComputeAndStoreDeltaBiasGPU(dZ, learning_rate, stream);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform bias update.");


    deltaW.allocateMemoryIfNotAllocatedZero(W.shape);
    ComputeAndStoreDeltaWeightsGPU(dZ, learning_rate, stream);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform weights update.");
    dZ = dA;
}

void LinearLayer::ComputeAndStoreDeltaBiasGPU(Matrix& dZ, float learning_rate, cudaStream_t stream) {
    dim3 block_size(1024);
    dim3 num_of_blocks( (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x);
    linearLayerComputeAndStoreDeltaBiasGPU<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
                                                         b.data_device.get(),
                                                         dZ.shape.x, dZ.shape.y,
                                                         b.shape.x, deltaBias.data_device.get(),
                                                         learning_rate);
}

void LinearLayer::ComputeAndStoreDeltaWeightsGPU(Matrix& dZ, float learning_rate, cudaStream_t stream) {
    dim3 block_size(16, 16);
    dim3 num_of_blocks(	(W.shape.x + block_size.x - 1) / block_size.x,
                           (W.shape.y + block_size.y - 1) / block_size.y);
    linearLayerComputeAndStoreDeltaWeightsGPU<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
                                                            A.data_device.get(),
                                                            W.data_device.get(),
                                                            dZ.shape.x, dZ.shape.y,
                                                            A.shape.x, A.shape.y,
                                                            deltaW.data_device.get(),
                                                            learning_rate);
}

void LinearLayer::computeAndStoreBackpropErrorGPU(Matrix& dZ, cudaStream_t stream) {
    dim3 block_size(16, 16);
    dim3 num_of_blocks(	(A.shape.x + block_size.x - 1) / block_size.x,
                           (A.shape.y + block_size.y - 1) / block_size.y);
    linearLayerBackpropGPU<<<num_of_blocks, block_size,0, stream>>>( W.data_device.get(),
                                                                     dZ.data_device.get(),
                                                                     dA.data_device.get(),
                                                                     W.shape.x, W.shape.y,
                                                                     dZ.shape.x, dZ.shape.y);
}


void LinearLayer::update() {

    updateBias(b.data_host.get(), deltaBias.data_host.get(), b.shape.x, b.shape.y);

    updateWeights(W.data_host.get(), deltaW.data_host.get(), W.shape.x, W.shape.y);
}

void LinearLayer::updateBiasGPU(){

    dim3 block_size(1024);
    dim3 num_of_blocks( (b.shape.y * b.shape.x + block_size.x - 1) / block_size.x);
    addMatrixDividedByScalar<<<num_of_blocks, block_size>>>(b.data_device.get(),
                                                                      deltaBias.data_device.get(),
                                                                      b.shape.x, b.shape.y, 1);

    deltaBias.setZeroDevice();
}

void LinearLayer::updateWeightsGPU(){

    dim3 block_size(1024);
    dim3 num_of_blocks( (W.shape.y * W.shape.x + block_size.x - 1) / block_size.x);
    addMatrixDividedByScalar<<<num_of_blocks, block_size>>>(W.data_device.get(),
                                                                      deltaW.data_device.get(),
                                                                      W.shape.x, W.shape.y, 1);
    deltaW.setZeroDevice();
}

void LinearLayer::accumulateDeltaGPU(NNLayer *layer, cudaStream_t stream){
    LinearLayer* linearLayer = static_cast<LinearLayer*>(layer);

    dim3 block_size(16, 16);

    linearLayer->deltaW.allocateMemoryIfNotAllocatedZero(W.shape);

    dim3 num_of_blocks(	(W.shape.x + block_size.x - 1) / block_size.x,
                           (W.shape.y + block_size.y - 1) / block_size.y);
    addMatrixDividedByScalar<<<num_of_blocks, block_size,0, stream>>>(
                                                         linearLayer->deltaW.data_device.get(),
                                                         deltaW.data_device.get(),
                                                         W.shape.x, W.shape.y, 1);
    deltaW.setZeroDevice();


    linearLayer->deltaBias.allocateMemoryIfNotAllocatedZero(b.shape);
    dim3 num_of_blocks2(	(b.shape.x + block_size.x - 1) / block_size.x,
                           (b.shape.y + block_size.y - 1) / block_size.y);

    addMatrixDividedByScalar<<<num_of_blocks2, block_size,0, stream>>>(
                                                        linearLayer->deltaBias.data_device.get(),
                                                        deltaBias.data_device.get(),
                                                        b.shape.x, b.shape.y, 1);
    deltaBias.setZeroDevice();


}

void LinearLayer::updateGPU() {
    updateBiasGPU();
    updateWeightsGPU();
}

Matrix& LinearLayer::backprop(Matrix& dZ, float learning_rate) {
    dA.allocateMemoryIfNotAllocated(A.shape);

    linearLayerBackprop(W.data_host.get(),
                        dZ.data_host.get(),
                        dA.data_host.get(),
                        W.shape.x, W.shape.y,
                        dZ.shape.x, dZ.shape.y);
    deltaW.allocateMemoryIfNotAllocatedZero(W.shape);

    linearLayerComputeAndStoreDeltaWeights(dZ.data_host.get(),
                                           A.data_host.get(),
                                           W.data_host.get(),
                                           dZ.shape.x, dZ.shape.y,
                                           A.shape.x, A.shape.y,
                                           deltaW.data_host.get(), learning_rate);

    deltaBias.allocateMemoryIfNotAllocatedZero(b.shape);
    linearLayerComputeAndStoreDeltaBias(dZ.data_host.get(),
                                        b.data_host.get(),
                                        dZ.shape.x, dZ.shape.y,
                                        b.shape.x, deltaBias.data_host.get(),
                                        learning_rate);
    return dA;
}

int LinearLayer::getXDim() const {
    return W.shape.x;
}

int LinearLayer::getYDim() const {
    return W.shape.y;
}

Matrix LinearLayer::getWeightsMatrix() const {
    return W;
}

Matrix LinearLayer::getBiasVector() const {
    return b;
}

void LinearLayer::updateBias(float* b, float * db, int b_x_dim, int b_y_dim) {
    for (int i = 0; i < b_x_dim * b_y_dim; ++i) {
        b[i] += db[i];
        db[i] = 0;
    }
}

void LinearLayer::updateWeights(float *W, float *dW, int W_x_dim, int W_y_dim) {
    for (int i = 0; i < W_x_dim * W_y_dim; ++i) {
        W[i] += dW[i];
        dW[i] = 0;
    }
}

NNLayer *LinearLayer::copy() const {
    LinearLayer* copyLayer = new LinearLayer(name, W.shape);

    copyLayer->W = Matrix(W);
    copyLayer->b = Matrix(b);
    copyLayer->deltaW = Matrix(deltaW);
    copyLayer->deltaBias = Matrix(deltaBias);
    copyLayer->Z = Matrix(Z);
    copyLayer->A = Matrix(A);
    copyLayer->dA = Matrix(dA);

    return copyLayer;
}




