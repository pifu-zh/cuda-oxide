/* [gfx1030 goal] attribute-number probe: validates the CUDA->HIP mapping fix
 * by querying HIP attributes BY NUMBER (enum values are the ABI).
 * Probes: 74 (MaxSharedMemoryPerBlock), 75 (SharedMemPerBlockOptin),
 *         76 (SharedMemPerMultiprocessor), 77 (SingleToDoublePrecisionPerfRatio),
 *         5 (MaxThreadsPerBlock), 23 (WarpSize) as sanity anchors.
 */
#include <dlfcn.h>
#include <stdio.h>
typedef int hipError_t;
int main(void) {
    void *h = dlopen("/opt/rocm/lib/libamdhip64.so.7", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    hipError_t (*hipInit)(unsigned) = dlsym(h, "hipInit");
    hipError_t (*hipGetDevice)(int *) = dlsym(h, "hipGetDevice");
    hipError_t (*hipDeviceGetAttribute)(int *, int, int) = dlsym(h, "hipDeviceGetAttribute");
    if (!hipInit || !hipGetDevice || !hipDeviceGetAttribute) {
        printf("dlsym failed\n"); return 1;
    }
    hipError_t rc = hipInit(0);
    printf("hipInit(0) -> %d\n", rc);
    int dev = -1;
    rc = hipGetDevice(&dev);
    printf("hipGetDevice -> %d (device %d)\n", rc, dev);
    int attrs[] = {74, 75, 76, 77, 5, 23};
    const char *names[] = {"74 MaxSharedMemoryPerBlock?", "75 SharedMemPerBlockOptin?",
                           "76 SharedMemPerMultiprocessor?", "77 SingleToDoublePerfRatio?",
                           "5 MaxThreadsPerBlock?", "23 WarpSize?"};
    for (unsigned i = 0; i < sizeof(attrs)/sizeof(attrs[0]); i++) {
        int v = -12345;
        hipError_t r = hipDeviceGetAttribute(&v, attrs[i], dev);
        printf("attr %-34s rc=%d value=%d\n", names[i], r, v);
    }
    return 0;
}
