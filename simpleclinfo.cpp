//a simple program that obtains the CL platform and devices, prints them out and exits

#include <array>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast.h>
#include <clblast_c.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            fprintf(stderr, "ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            fprintf(stderr, "You may be out of VRAM. Please check if you have enough.\n");\
            exit(1);                                                \
        }                                                           \
    } while (0)

static cl_platform_id platform;
static cl_device_id device;

int main(void) {

    cl_int err;

    struct cl_device;
    struct cl_platform {
        cl_platform_id id;
        unsigned number;
        char name[128];
        char vendor[128];
        struct cl_device * devices;
        unsigned n_devices;
        struct cl_device * default_device;
    };

    struct cl_device {
        struct cl_platform * platform;
        cl_device_id id;
        unsigned number;
        cl_device_type type;
        char name[128];
        cl_ulong global_mem_size;
    };

    enum { NPLAT = 16, NDEV = 16 };

    struct cl_platform platforms[NPLAT];
    unsigned n_platforms = 0;
    struct cl_device devices[NDEV];
    unsigned n_devices = 0;
    struct cl_device * default_device = NULL;

    platform = NULL;
    device = NULL;

    cl_platform_id platform_ids[NPLAT];
    CL_CHECK(clGetPlatformIDs(NPLAT, platform_ids, &n_platforms));

    std::string output = "{\"devices\":[";

    for (unsigned i = 0; i < n_platforms; i++) {
        struct cl_platform * p = &platforms[i];
        p->number = i;
        p->id = platform_ids[i];
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_NAME, sizeof(p->name), &p->name, NULL));
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_VENDOR, sizeof(p->vendor), &p->vendor, NULL));

        cl_device_id device_ids[NDEV];
        cl_int clGetDeviceIDsError = clGetDeviceIDs(p->id, CL_DEVICE_TYPE_ALL, NDEV, device_ids, &p->n_devices);
        if (clGetDeviceIDsError == CL_DEVICE_NOT_FOUND) {
            p->n_devices = 0;
        } else {
            CL_CHECK(clGetDeviceIDsError);
        }
        p->devices = p->n_devices > 0 ? &devices[n_devices] : NULL;
        p->default_device = NULL;

        std::string platformtemplate = "{\"online\":[";

        for (unsigned j = 0; j < p->n_devices; j++) {
            struct cl_device * d = &devices[n_devices];
            d->number = n_devices++;
            d->id = device_ids[j];
            d->platform = p;
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_NAME, sizeof(d->name), &d->name, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_TYPE, sizeof(d->type), &d->type, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(d->global_mem_size), &d->global_mem_size, NULL));
            std::string devicetemplate = "{\"CL_DEVICE_NAME\":\"" + std::string(d->name) + "\", \"CL_DEVICE_GLOBAL_MEM_SIZE\":"+std::to_string(d->global_mem_size)+"}";
            if(j>0)
            {
                devicetemplate = ","+devicetemplate;
            }
            platformtemplate += devicetemplate;
        }

        platformtemplate += "]}";
        if(i>0)
        {
            platformtemplate = ","+platformtemplate;
        }
        output += platformtemplate;
    }
    output += "]}";
    printf("%s",output.c_str());
    fflush(stdout);
    return 0;
}
