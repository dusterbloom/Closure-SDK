#ifndef CLOSURE_KERNEL_H
#define CLOSURE_KERNEL_H

#include <stdint.h>

#define VOCAB_SIZE     64
#define GENOME_MAX     5000
#define KNN_K          5
#define NUM_CHANNELS   2
#define THRESHOLD      0.7854f
#define SLERP_SLOW     0.05f
#define SLERP_FAST     0.3f

typedef struct { float w, x, y, z; } Quat;

typedef struct {
    Quat context;
    Quat observation;
} GenomeEntry;

typedef struct {
    Quat cell_c[NUM_CHANNELS];
    GenomeEntry genome[GENOME_MAX];
    Quat operators[VOCAB_SIZE];
    uint16_t genome_count;
    float distances[GENOME_MAX];
    uint16_t indices[GENOME_MAX];
} ClosureKernel;

void kernel_init(ClosureKernel *k);
Quat process_token(ClosureKernel *k, uint8_t token);
int closure_check(Quat predicted, Quat observed);

#endif
