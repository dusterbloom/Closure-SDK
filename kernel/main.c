#include "closure_kernel.h"
#include <stdio.h>
#include <math.h>

#define REPORT_INTERVAL 100

static int quat_closest_token(ClosureKernel *k, Quat q) {
    int best = 0;
    float best_dist = 1e9f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        float d = fabsf(k->operators[i].w * q.w +
                        k->operators[i].x * q.x +
                        k->operators[i].y * q.y +
                        k->operators[i].z * q.z);
        float dist = (d > 1.0f) ? 0.0f : acosf(d);
        if (dist < best_dist) { best_dist = dist; best = i; }
    }
    return best;
}

int main(void) {
    static ClosureKernel k;
    kernel_init(&k);

    int c;
    long token_count = 0;
    long correct_total = 0;
    long correct_window = 0;
    int ring[REPORT_INTERVAL];
    int ring_pos = 0;

    for (int i = 0; i < REPORT_INTERVAL; i++) ring[i] = 0;

    if ((c = getchar()) == EOF) {
        printf("Closure Kernel initialized. Genome: 0 entries.\n");
        return 0;
    }

    do {
        uint8_t tok;
        if (c >= 32 && c <= 126)
            tok = (uint8_t)((c - 32) % VOCAB_SIZE);
        else
            tok = 0;

        Quat pred = process_token(&k, tok);
        int pred_tok = quat_closest_token(&k, pred);
        int hit = (pred_tok == (int)tok) ? 1 : 0;

        correct_window -= ring[ring_pos];
        ring[ring_pos] = hit;
        correct_window += hit;
        correct_total += hit;
        ring_pos = (ring_pos + 1) % REPORT_INTERVAL;
        token_count++;

        if (token_count % REPORT_INTERVAL == 0) {
            float acc = (float)correct_window / (float)REPORT_INTERVAL * 100.0f;
            printf("tokens=%ld genome=%u last%d_acc=%.1f%%\n",
                   token_count, k.genome_count, REPORT_INTERVAL, acc);
        }
    } while ((c = getchar()) != EOF);

    if (token_count > 0) {
        float overall_acc = (float)correct_total / (float)token_count * 100.0f;
        float compression = (float)k.genome_count / (float)token_count;
        printf("\n--- Final Stats ---\n");
        printf("Total tokens  : %ld\n", token_count);
        printf("Genome size   : %u\n", k.genome_count);
        printf("Top-1 accuracy: %.2f%%\n", overall_acc);
        printf("Compression   : %.4f (genome/tokens)\n", compression);
    } else {
        printf("Closure Kernel initialized. Genome: 0 entries.\n");
    }

    return 0;
}
