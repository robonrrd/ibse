#ifndef _IBSE_COMMON_H_
#define _IBSE_COMMON_H_

namespace ibse
{

enum class ErrorCode
{
    SUCCESS,

    // Setting data
    ERROR_INCONSISTENT_DATA_SIZES,
    ERROR_NONMONOTONIC_TIMES,
      
    // Initial alignment estimation
    ERROR_NO_DATA,
      
    // Scale estimation
    ERROR_NO_INITIAL_ALIGNMENT,
    ERROR_FINAL_ESTIMATION_FAILED
};

}; // namespace ibse

#endif
