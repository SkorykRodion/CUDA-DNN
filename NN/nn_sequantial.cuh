#ifndef COURSE_TRY_2_NN_SEQUANTIAL_CUH
#define COURSE_TRY_2_NN_SEQUANTIAL_CUH


#include <vector>
#include "../layers/nn_layer.cuh"
#include "../utils/bce_cost.cuh"
#include "../dataset/coordinates_dataset.cuh"
#include "nn_abstract.cuh"
#include "../utils/nn_exeption.cuh"
#include "../dataset/coordinates_dataset.cuh"
#include <chrono>

class NeuralNetwork{ //: public NeuralNetwork
private:
    std::vector<NNLayer*> layers;
    BCECost bce_cost;

    Matrix Y;
    Matrix dY;
    float learning_rate;

public:
    NeuralNetwork(float learning_rate = 0.01);
    ~NeuralNetwork();

    Matrix forward(Matrix X);
    void backprop(Matrix predictions, Matrix target);

    void addLayer(NNLayer *layer);
    std::vector<NNLayer*> getLayers() const;

    void fit(Dataset& dataset, unsigned int epochs);

    static float computeAccuracy(const Matrix& predictions, const Matrix& targets);

    void update();
};

#endif //COURSE_TRY_2_NN_SEQUANTIAL_CUH
