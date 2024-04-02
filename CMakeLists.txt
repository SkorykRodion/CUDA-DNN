cmake_minimum_required(VERSION 3.27)
project(course_try_2 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(course_try_2 main.cu
        NN/nn_abstract.cu
        NN/nn_abstract.cuh
        utils/matrix.cu
        utils/matrix.cuh
        utils/shape.cu
        utils/shape.cuh
        utils/nn_exeption.cu
        utils/nn_exeption.cuh
        utils/cost.cu
        utils/cost.cuh
        layers/nn_layer.cu
        layers/nn_layer.cuh
        dataset/dataset.cu
        dataset/dataset.cuh
        layers/sigmoid_activation.cu
        layers/sigmoid_activation.cuh
        layers/sigmoid_activation.cuh
        layers/relu_activation.cu
        layers/relu_activation.cuh
        layers/linear_layer.cu
        layers/linear_layer.cuh
        layers/linear_layer.cuh
        utils/bce_cost.cu
        utils/bce_cost.cuh
        NN/nn_sequantial.cu
        NN/nn_sequantial.cuh
        dataset/coordinates_dataset.cu
        dataset/coordinates_dataset.cuh
        NN/nn_parallel.cu
        NN/nn_parallel.cuh
)

set_target_properties(course_try_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
