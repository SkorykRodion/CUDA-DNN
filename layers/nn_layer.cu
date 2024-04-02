#include "nn_layer.cuh"
void NNLayer::setHasWeights(bool hasWeights) {
    NNLayer::hasWeights = hasWeights;
}

bool NNLayer::isHasWeights() const {
    return hasWeights;
}
