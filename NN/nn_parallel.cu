
#include "nn_parallel.cuh"

void NeuralNetworkParallel::deleteLayers(const std::vector<std::vector<NNLayer*>>& layers) {
    for (const auto& layerVector : layers) {
        for (auto layer : layerVector) {
            delete layer;
        }
    }
}

NeuralNetworkParallel::NeuralNetworkParallel(float learning_rate) :
        learning_rate(learning_rate)
{ }

NeuralNetworkParallel::~NeuralNetworkParallel() {
    for (auto layer : layers) {
        delete layer;
    }
    for (auto stream : streams) {
        cudaStreamDestroy(stream);
    }
}

void NeuralNetworkParallel::addLayer(NNLayer *layer) {
    this->layers.push_back(layer);
}

std::vector<NNLayer*> NeuralNetworkParallel::getLayers() const {
    return layers;
}

float NeuralNetworkParallel::computeAccuracy(const Matrix &predictions, const Matrix &targets) {
    int m = predictions.shape.x;
    int correct_predictions = 0;

    for (int i = 0; i < m; i++) {
        float prediction = predictions[i] > 0.5 ? 1 : 0;
        if (prediction == targets[i]) {
            correct_predictions++;
        }
    }
    return 1.*correct_predictions/m;
}

void NeuralNetworkParallel::fit(Dataset &dataset, unsigned int epochs) {
    auto start_time = std::chrono::high_resolution_clock::now();
    createStreams(dataset.getNumOfBatches());
    preds.resize(dataset.getNumOfBatches());
    for (int epoch = 0; epoch < epochs; epoch++) {
        float cost = 0;
        auto layersForBatches = copyLayersForBatches(getLayers(), dataset.getNumOfBatches());
        for (int batch = 0; batch < dataset.getNumOfBatches(); batch++) {
            preds[batch] = forward(dataset.getBatches().at(batch),
                        layersForBatches[batch], streams[batch]);

            backprop(preds[batch], dataset.getTargets().at(batch),
                     layersForBatches[batch], streams[batch]);
        }

        for (int batch = 0; batch < dataset.getNumOfBatches(); batch++) {
            auto& layersForBatch = layersForBatches[batch];
            for (int i = 0; i < layersForBatch.size(); ++i) {
                auto layer = layersForBatch[i];
                if (layer->isHasWeights()) {
                    layer->accumulateDeltaGPU(layers[i],streams[batch]);
                }
            }
        }
        cudaDeviceSynchronize();
        update();
        for (int batch = 0; batch < dataset.getNumOfBatches(); batch++) {
            cudaDeviceSynchronize();
            cost += bce_cost.cost(preds[batch],
                                  dataset.getTargets().at(batch));
        }
        deleteLayers(layersForBatches);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        if (epoch % 1 == 0) {
            std::cout 	<< "Epoch: " << epoch+1
                         << ", Cost: " << cost/dataset.getNumOfBatches()
                         << std::endl<< "Elapsed time for fit function: "
                         << elapsed_seconds.count() << " seconds." << std::endl;;
        }
    }
}




void NeuralNetworkParallel::copyLayers(const std::vector<NNLayer *> &source, std::vector<NNLayer *> &destination) {
    for (NNLayer* layer : source) {
        destination.push_back(layer->copy());
    }
}

std::vector<std::vector<NNLayer*>> NeuralNetworkParallel::copyLayersForBatches(const std::vector<NNLayer*>& source, size_t numBatches) {
    std::vector<std::vector<NNLayer*>> layersPerBatch(numBatches);
    for (size_t i = 0; i < numBatches; ++i) {
        copyLayers(source, layersPerBatch[i]);
    }
    return layersPerBatch;
}

Matrix NeuralNetworkParallel::forward(Matrix X, std::vector<NNLayer *> batchLayers, cudaStream_t stream) {
    Matrix Z(X);
    for (auto layer : batchLayers) {
        layer->forwardGPU(Z, stream);
    }
    return Z;
}

void NeuralNetworkParallel::backprop(Matrix predictions, Matrix target, std::vector<NNLayer *> batchLayers,
                                     cudaStream_t stream) {
    Matrix dY;
    dY.allocateMemoryIfNotAllocated(predictions.shape);
    Matrix error = bce_cost.dCostGPU(predictions, target, dY, stream);

    for (auto it = batchLayers.rbegin(); it != batchLayers.rend(); it++) {
        (*it)->backpropGPU(dY, learning_rate, stream);
    }

}


void NeuralNetworkParallel::createStreams(size_t numBatches) {
    streams.resize(numBatches);
    for (size_t i = 0; i < numBatches; ++i) {
        cudaStreamCreate(&streams[i]);
    }
}

void NeuralNetworkParallel::update() {
    for (auto layer : layers) {
        if(layer->isHasWeights()){
            layer->updateGPU();
        }
    }
}
