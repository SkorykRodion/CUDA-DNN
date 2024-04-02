#ifndef COURSE_TRY_2_NN_ABSTRACT_CUH
#define COURSE_TRY_2_NN_ABSTRACT_CUH
#include <vector>
#include "../layers/nn_layer.cuh"
#include "../utils/cost.cuh"
#include "../utils/matrix.cuh"
#include "../dataset/dataset.cuh"

class AbstractNeuralNetwork {
protected:
    std::vector<NNLayer*> layers;
    Cost* cost;

    Matrix Y;
    Matrix dY;
    float learning_rate;

public:


    virtual Matrix forward(Matrix X) = 0;
    virtual void backprop(Matrix predictions, Matrix target) = 0;
    virtual void addLayer(NNLayer* layer) = 0;
    virtual std::vector<NNLayer*> getLayers() const = 0;
    virtual void fit(Dataset& dataset, unsigned int epochs) =0;
    static float computeAccuracy(const Matrix& predictions, const Matrix& targets);
};


#endif //COURSE_TRY_2_NN_ABSTRACT_CUH
