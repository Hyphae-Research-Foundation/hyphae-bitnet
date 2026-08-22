/* DRAM bandwidth. Arrays >> cache. GB/s is bus traffic, not memcpy throughput.
 *
 *   copy  B=A              2 bytes/elem
 *   scale B=s*A            2
 *   add   C=A+B            3
 *   triad C=A+s*B          3   (STREAM)
 *   read  acc+=A           1   (weight-fetch shape)
 *
 * Usage: measure-dram [GiB_per_array] [threads] [reps]
 */
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *alloc_pages(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 4096, bytes) != 0) return NULL;
    memset(p, 1, bytes);
    return p;
}

int main(int argc, char **argv) {
    const size_t gib = (argc > 1) ? (size_t)atoi(argv[1]) : 8;
    const int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    const int reps = (argc > 3) ? atoi(argv[3]) : 5;
    if (gib < 1 || threads < 1 || reps < 1) return 2;

    const size_t n = (gib << 30) / sizeof(double);
    const size_t bytes = n * sizeof(double);
    omp_set_num_threads(threads);

    double *a = alloc_pages(bytes);
    double *b = alloc_pages(bytes);
    double *c = alloc_pages(bytes);
    if (!a || !b || !c) {
        fprintf(stderr, "alloc failed for 3 x %zu GiB\n", gib);
        return 1;
    }

    const double scalar = 1.0001;
    volatile double sink = 0.0;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    printf("# graviton dram  array=%zuGiB  threads=%d  reps=%d  n=%zu\n",
           gib, threads, reps, n);
    printf("%-8s %10s %12s\n", "kernel", "best_s", "gbs_traffic");

    for (int k = 0; k < 5; k++) {
        double best = 1e99;
        for (int r = 0; r < reps; r++) {
            const double t0 = now_s();
            if (k == 0) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) b[i] = a[i];
            } else if (k == 1) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) b[i] = scalar * a[i];
            } else if (k == 2) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
            } else if (k == 3) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) c[i] = a[i] + scalar * b[i];
            } else {
                double acc = 0.0;
                #pragma omp parallel for schedule(static) reduction(+ : acc)
                for (size_t i = 0; i < n; i++) acc += a[i];
                sink += acc;
            }
            const double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        static const char *names[] = {"copy", "scale", "add", "triad", "read"};
        static const int traffic[] = {2, 2, 3, 3, 1};
        const double gbs = (traffic[k] * (double)bytes / 1e9) / best;
        printf("%-8s %10.4f %12.2f\n", names[k], best, gbs);
    }
    if (sink == 0.0) fprintf(stderr, "unexpected empty read\n");
    free(a);
    free(b);
    free(c);
    return 0;
}
