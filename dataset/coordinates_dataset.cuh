#ifndef COURSE_TRY_2_COORDINATES_DATASET_CUH
#define COURSE_TRY_2_COORDINATES_DATASET_CUH


#include "../utils/matrix.cuh"
#include "dataset.cuh"

#include <vector>

class CoordinatesDataset : public Dataset{
private:
    size_t batch_size;
    size_t number_of_batches;

    std::vector<Matrix> batches;
    std::vector<Matrix> targets;

public:

    CoordinatesDataset(size_t batch_size, size_t number_of_batches);

    int getNumOfBatches();
    std::vector<Matrix>& getBatches();
    std::vector<Matrix>& getTargets();

};


#endif //COURSE_TRY_2_COORDINATES_DATASET_CUH
