#include "nn_sequantial.cuh"

NeuralNetwork::NeuralNetwork(float learning_rate) :
        learning_rate(learning_rate)
{ }

NeuralNetwork::~NeuralNetwork() {
    for (auto layer : layers) {
        delete layer;
    }
}

void NeuralNetwork::addLayer(NNLayer* layer) {
    this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
    Matrix Z = X;

    for (auto layer : layers) {
        Z = layer->forward(Z);
    }

    Y = Z;
    return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) {
    dY.allocateMemoryIfNotAllocated(predictions.shape);
    Matrix error = bce_cost.dCost(predictions, target, dY);
    cudaDeviceSynchronize();
    error.copyDeviceToHost();

    for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
        error = (*it)->backprop(error, learning_rate);
    }
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
    return layers;
}

void NeuralNetwork::update(){
    for (auto layer : layers) {
        if(layer->isHasWeights()){
            layer->update();
        }
    }
}

void NeuralNetwork::fit(Dataset& dataset, unsigned int epochs) {
    auto start_time = std::chrono::high_resolution_clock::now();
    Matrix Y;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float cost = 0.0;

        for (int batch = 0; batch < dataset.getNumOfBatches(); batch++) {
            Y = forward(dataset.getBatches().at(batch));
            Y.copyHostToDevice();
            backprop(Y, dataset.getTargets().at(batch));
            cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
        }
        update();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        if (epoch % 1 == 0) {
            std::cout 	<< "Epoch: " << epoch+1
                         << ", Cost: " << cost / dataset.getNumOfBatches()
                         << std::endl << "Elapsed time for fit function: "
                         << elapsed_seconds.count() << " seconds." << std::endl;;
        }
    }
}

float NeuralNetwork::computeAccuracy(const Matrix &predictions, const Matrix &targets) {
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