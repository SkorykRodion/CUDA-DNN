#include "nn_exeption.cuh"

const char *NNException::what() const throw() {
    return exception_message;
}

NNException::NNException(const char *exception_message) :
        exception_message(exception_message)
{ }

void NNException::throwIfDeviceErrorsOccurred(const char *exception_message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << error << ": " << exception_message;
        throw NNException(exception_message);
    }
}
