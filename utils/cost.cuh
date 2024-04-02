#ifndef COURSE_TRY_2_COST_CUH
#define COURSE_TRY_2_COST_CUH

#include "matrix.cuh"

class Cost {
public:
    virtual float cost(Matrix predictions, Matrix target) = 0;
    virtual Matrix dCost(Matrix predictions, Matrix target, Matrix dY) = 0;
};


#endif //COURSE_TRY_2_COST_CUH
