/*
<%
import os
import sys
if os.name == 'nt':
    # Clean up environment variables that might have been set incorrectly
    for env_var in ['CFLAGS', 'CXXFLAGS']:
        if env_var in os.environ:
            os.environ[env_var] = os.environ[env_var].replace('-std=c++11', '')
            os.environ[env_var] = os.environ[env_var].replace('-fvisibility=hidden', '')
    cfg['compiler_args'] = ['/std:c++14', '/EHsc']
else:
    args = ['-std=c++11']
    if sys.platform == 'darwin':
        args.extend(['-undefined', 'dynamic_lookup'])
    cfg['compiler_args'] = args
%>
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <ctime>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

std::mt19937 generator;

int randint_(int end)
{
    std::uniform_int_distribution<int> dist(0, end - 1);
    return dist(generator);
}

py::array_t<int> sample_negative(
    int userNum, int itemNum, int trainNum, 
    vector<vector<int>> allPos, int neg_ratio, py::array_t<double> neg_weights
)
{
    int samplesPerUser = (trainNum / userNum);
    int row = neg_ratio + 2;

    py::array_t<int> S_array = py::array_t<int>({userNum * samplesPerUser, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    std::discrete_distribution<int> discrete_dist;
    std::uniform_int_distribution<int> uniform_dist(0, itemNum - 1); // Inicjalizacja uniform_dist tutaj
    bool use_discrete_sampling = false;

    if (neg_weights.size() != 0)
    {
        py::buffer_info weights_buf = neg_weights.request();
        double* weights_ptr = (double*) weights_buf.ptr;
        discrete_dist = std::discrete_distribution<int>(weights_ptr, weights_ptr + itemNum);
        use_discrete_sampling = true;
    }

    for (int user = 0; user < userNum; user++)
    {
        vector<int> userPos = allPos[user];

        for (int pair_i = 0; pair_i < samplesPerUser; pair_i++)
        {
            int posIndex = randint_((int)userPos.size());
            int posItem = userPos[posIndex];
            ptr[(user * samplesPerUser + pair_i) * row] = user;
            ptr[(user * samplesPerUser + pair_i) * row + 1] = posItem;

            int negItem = 0;
            
            for (int index = 2; index < neg_ratio + 2; index++)
            {
                do
                {
                    if (use_discrete_sampling)
                    {
                        negItem = discrete_dist(generator);
                    }
                    else
                    {
                        negItem = uniform_dist(generator);
                    }
                }
                while (find(userPos.begin(), userPos.end(), negItem) != userPos.end());

                ptr[(user * samplesPerUser + pair_i) * row + index] = negItem;
            }
        }
    }
    return S_array;
}

void set_seed(unsigned int seed)
{
    generator.seed(seed);
}

using namespace py::literals;

PYBIND11_MODULE(sampling, m)
{
    set_seed((unsigned int)time(nullptr)); // Użycie time(nullptr) dla lepszej praktyki
    m.doc() = "example plugin";
    m.def("seed", &set_seed, "set random seed", "seed"_a);
    m.def(
        "sample_negative", &sample_negative, "sampling negatives for all", 
        "userNum"_a, "itemNum"_a, "trainNum"_a, "allPos"_a, "neg_ratio"_a, "neg_weights"_a
    );
}