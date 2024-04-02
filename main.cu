#include <iostream>
#include <time.h>


#include "layers/linear_layer.cuh"
#include "layers/relu_activation.cuh"
#include "layers/sigmoid_activation.cuh"

#include <cuda_runtime.h>
#include "dataset/coordinates_dataset.cuh"
#include "utils/bce_cost.cuh"
#include "NN/nn_sequantial.cuh"
#include "NN/nn_parallel.cuh"
#include <cuda_runtime.h>



int main() {
    srand( 0 ); //time(NULL)

    CoordinatesDataset dataset(5000, 1);
    std::cout<<"start\n";
    std::cout<<"Parallel\n";
    NeuralNetworkParallel nn1;
    nn1.addLayer(new LinearLayer("linear_1", Shape(2, 500)));
    nn1.addLayer(new ReLUActivation("relu_1"));
    nn1.addLayer(new LinearLayer("linear_2", Shape(500, 1)));
    nn1.addLayer(new SigmoidActivation("sigmoid_output"));
    nn1.fit(dataset, 1);
    std::cout<<"Sequential\n";
    NeuralNetwork nn2;
    nn2.addLayer(new LinearLayer("linear_1", Shape(2, 500)));
    nn2.addLayer(new ReLUActivation("relu_1"));
    nn2.addLayer(new LinearLayer("linear_2", Shape(500, 1)));
    nn2.addLayer(new SigmoidActivation("sigmoid_output"));
    nn2.fit(dataset, 1);


return 0;

}

