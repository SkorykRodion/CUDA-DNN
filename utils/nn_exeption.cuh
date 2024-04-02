#ifndef COURSE_TRY_2_NN_EXEPTION_CUH
#define COURSE_TRY_2_NN_EXEPTION_CUH


#include <exception>
#include <iostream>

class NNException : std::exception {
private:
    const char* exception_message;

public:
    NNException(const char* exception_message);

    virtual const char* what() const throw();

    static void throwIfDeviceErrorsOccurred(const char* exception_message);
};


#endif //COURSE_TRY_2_NN_EXEPTION_CUH
