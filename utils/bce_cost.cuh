#ifndef COURSE_TRY_2_BCE_COST_CUH
#define COURSE_TRY_2_BCE_COST_CUH


#include <driver_types.h>
#include "matrix.cuh"
#include "nn_exeption.cuh"
#include <cassert>

class BCECost {
public:
    float cost(Matrix predictions, Matrix target);
    Matrix dCost(Matrix predictions, Matrix target, Matrix dY);

    Matrix dCostGPU(Matrix predictions, Matrix target, Matrix& dY, cudaStream_t stream);
};


#endif //COURSE_TRY_2_BCE_COST_CUH
