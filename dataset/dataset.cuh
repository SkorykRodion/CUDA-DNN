#ifndef COURSE_TRY_2_DATASET_CUH
#define COURSE_TRY_2_DATASET_CUH


#include <vector>
#include "../utils/matrix.cuh" // Assuming Matrix is defined elsewhere

class Dataset {
protected:
    size_t batch_size;
    size_t number_of_batches;

    std::vector<Matrix> batches;
    std::vector<Matrix> targets;

public:
    Dataset() : batch_size(1), number_of_batches(1) {}

    Dataset(size_t batch_size, size_t number_of_batches) :
            batch_size(batch_size), number_of_batches(number_of_batches) {}

    virtual ~Dataset() {}

    virtual int getNumOfBatches()= 0;
    virtual std::vector<Matrix>& getBatches() = 0;
    virtual std::vector<Matrix>& getTargets() = 0;
};

#endif //COURSE_TRY_2_DATASET_CUH
