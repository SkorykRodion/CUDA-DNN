#ifndef COURSE_TRY_2_NN_LAYER_CUH
#define COURSE_TRY_2_NN_LAYER_CUH
#include <iostream>
#include "../utils/matrix.cuh"

class NNLayer {
protected:
    std::string name;
    bool hasWeights;
public:
    virtual ~NNLayer() = 0;

    virtual Matrix& forward(Matrix& A) = 0;
    virtual Matrix& backprop(Matrix& dZ, float learning_rate) = 0;
    virtual NNLayer* copy() const = 0;
    virtual void forwardGPU(Matrix& Z, cudaStream_t stream) = 0;
    virtual void backpropGPU(Matrix& dZ, float learning_rate, cudaStream_t stream) = 0;
    bool isHasWeights() const;

    void setHasWeights(bool hasWeights);

    virtual void update() = 0;

    virtual void updateGPU() = 0;

    virtual void accumulateDeltaGPU(NNLayer *layer, cudaStream_t stream)=0;

    std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}


#endif //COURSE_TRY_2_NN_LAYER_CUH
