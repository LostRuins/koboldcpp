#include "ggml.h"
#include "common.h"
#include "mtmd/clip.h"
#include "mtmd/llava.h"
#include "llama.h"

#include "base64.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>


int main(int argc, char ** argv) {
    ggml_time_init();

     if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s mmproj-f16.gguf output-mmproj-quantized.gguf TYPE\n", argv[0]);
        printf("\nGGML_TYPE_F16    = 1\nGGML_TYPE_Q4_0    = 2\nGGML_TYPE_Q4_1    = 3\nGGML_TYPE_Q5_0    = 6\nGGML_TYPE_Q5_1    = 7\nGGML_TYPE_Q8_0    = 8\n");

        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    int type = GGML_TYPE_Q4_1;

    if(argc==4)
    {
        type = std::stoi(argv[3]);
    }

    printf("quantizing mmproj clip model to type=%d... ",type);
    clip_model_quantize(fname_inp.c_str(), fname_out.c_str(), type);
    printf("\ndone\n");

    return 0;
}
