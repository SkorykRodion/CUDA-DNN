#ifndef COURSE_TRY_2_NN_PARALLEL_CUH
#define COURSE_TRY_2_NN_PARALLEL_CUH


#include <vector>
#include <driver_types.h>
#include "../layers/nn_layer.cuh"
#include "../utils/bce_cost.cuh"
#include "../dataset/coordinates_dataset.cuh"
#include "nn_abstract.cuh"
#include <chrono>

class NeuralNetworkParallel{
private:
    std::vector<NNLayer*> layers;
    BCECost bce_cost;
    std::vector<cudaStream_t> streams;
    Matrix Y;
    Matrix dY;
    std::vector<Matrix> preds;
    float learning_rate;



public:
    void copyLayers(const std::vector<NNLayer*>& source, std::vector<NNLayer*>& destination);
    std::vector<std::vector<NNLayer*>> copyLayersForBatches(const std::vector<NNLayer*>& source, size_t numBatches);
    NeuralNetworkParallel(float learning_rate = 0.01);
    ~NeuralNetworkParallel();
    Matrix forward(Matrix X, std::vector<NNLayer*> batchLayers, cudaStream_t stream);

    void backprop(Matrix predictions, Matrix target,  std::vector<NNLayer*> batchLayers, cudaStream_t stream);

    void addLayer(NNLayer *layer);
    std::vector<NNLayer*> getLayers() const;

    void fit(Dataset& dataset, unsigned int epochs);

    static float computeAccuracy(const Matrix& predictions, const Matrix& targets);

    void update();


    void createStreams(size_t numBatches);

    void deleteLayers(const std::vector<std::vector<NNLayer *>> &layers);
};


#endif //COURSE_TRY_2_NN_PARALLEL_CUH
