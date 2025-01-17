#ifndef NNET_BERNOULLI_H_
#define NNET_BERNOULLI_H_

#include "nnet_common.h"

namespace nnet {

struct bernoulli_config {
    static const unsigned n_in = 10;
    const double thr = 0.5;
};

template<class data_T, typename CONFIG_T>
void bernoulli(
    data_T input[CONFIG_T::n_in],
    data_T output[CONFIG_T::n_in]
) {
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        output[i] = input[i];
        // if (input[i] < CONFIG_T::thr)
        //     output[i] = 0;
        // else
        //     output[i] = 1;
    }
}

}

#endif
