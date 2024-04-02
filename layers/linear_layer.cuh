#ifndef COURSE_TRY_2_LINEAR_LAYER_CUH
#define COURSE_TRY_2_LINEAR_LAYER_CUH

#include <driver_types.h>
#include "nn_layer.cuh"
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <random>
#include "linear_layer.cuh"
#include "../utils/nn_exeption.cuh"

class LinearLayer : public NNLayer {
private:
    const float weights_init_threshold = 0.01;

    Matrix W;
    Matrix b;
    Matrix deltaW;
    Matrix deltaBias;

private:

    Matrix Z;
    Matrix A;
    Matrix dA;

    void initializeBiasWithZeros();
    void initializeWeightsRandomly();

public:
    LinearLayer(std::string name, Shape W_shape);
    ~LinearLayer();

    Matrix& forward(Matrix& A);
    Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);
    int getXDim() const;
    int getYDim() const;

    Matrix getWeightsMatrix() const;
    Matrix getBiasVector() const;

    NNLayer* copy() const;

    void update();

    void updateBias(float *b, float *db, int b_x_dim, int b_y_dim);

    void updateWeights(float *W, float *dW, int W_x_dim, int W_y_dim);

    void forwardGPU(Matrix& A, cudaStream_t stream);

    void backpropGPU(Matrix& dZ, float learning_rate, cudaStream_t stream);

    void computeAndStoreBackpropErrorGPU(Matrix &dZ, cudaStream_t stream);

    void ComputeAndStoreDeltaWeightsGPU(Matrix &dZ, float learning_rate, cudaStream_t stream);

    void ComputeAndStoreDeltaBiasGPU(Matrix &dZ, float learning_rate, cudaStream_t stream);

    void updateBiasGPU();

    void updateWeightsGPU();

    void updateGPU();

    void accumulateDeltaGPU(NNLayer *layer, cudaStream_t stream);
};


#endif //COURSE_TRY_2_LINEAR_LAYER_CUH
