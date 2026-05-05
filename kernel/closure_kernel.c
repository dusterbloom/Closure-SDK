#include "closure_kernel.h"
#include <math.h>
#include <string.h>

static inline Quat quat_mul(Quat a, Quat b) {
    Quat r;
    r.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
    r.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
    r.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
    r.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
    return r;
}

static inline Quat quat_normalize(Quat q) {
    float n = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (n < 1e-9f) { q.w = 1.0f; q.x = q.y = q.z = 0.0f; return q; }
    float inv = 1.0f / n;
    q.w *= inv; q.x *= inv; q.y *= inv; q.z *= inv;
    return q;
}

static inline float quat_dot(Quat a, Quat b) {
    return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline float geodesic_dist(Quat a, Quat b) {
    float d = fabsf(quat_dot(a, b));
    if (d > 1.0f) d = 1.0f;
    return acosf(d);
}

static Quat quat_slerp(Quat a, Quat b, float t) {
    float dot = quat_dot(a, b);
    if (dot < 0.0f) {
        b.w = -b.w; b.x = -b.x; b.y = -b.y; b.z = -b.z;
        dot = -dot;
    }
    if (dot > 0.9995f) {
        Quat r;
        r.w = a.w + t*(b.w - a.w);
        r.x = a.x + t*(b.x - a.x);
        r.y = a.y + t*(b.y - a.y);
        r.z = a.z + t*(b.z - a.z);
        return quat_normalize(r);
    }
    float theta0 = acosf(dot);
    float theta = theta0 * t;
    float sin_theta = sinf(theta);
    float sin_theta0 = sinf(theta0);
    float s0 = cosf(theta) - dot * sin_theta / sin_theta0;
    float s1 = sin_theta / sin_theta0;
    Quat r;
    r.w = s0*a.w + s1*b.w;
    r.x = s0*a.x + s1*b.x;
    r.y = s0*a.y + s1*b.y;
    r.z = s0*a.z + s1*b.z;
    return quat_normalize(r);
}

static Quat knn_predict(ClosureKernel *k, Quat query) {
    uint16_t n = k->genome_count;
    if (n == 0) {
        Quat identity = {1.0f, 0.0f, 0.0f, 0.0f};
        return identity;
    }

    for (uint16_t i = 0; i < n; i++) {
        k->distances[i] = geodesic_dist(k->genome[i].context, query);
        k->indices[i] = i;
    }

    int kk = (n < KNN_K) ? n : KNN_K;
    for (int i = 0; i < kk; i++) {
        int min_j = i;
        for (uint16_t j = (uint16_t)(i + 1); j < n; j++) {
            if (k->distances[k->indices[j]] < k->distances[k->indices[min_j]])
                min_j = j;
        }
        uint16_t tmp = k->indices[i];
        k->indices[i] = k->indices[min_j];
        k->indices[min_j] = tmp;
    }

    float acc_w = 0.0f, acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;
    float total_weight = 0.0f;
    for (int i = 0; i < kk; i++) {
        float d = k->distances[k->indices[i]];
        float w = 1.0f / (d + 1e-6f);
        Quat obs = k->genome[k->indices[i]].observation;
        acc_w += w * obs.w;
        acc_x += w * obs.x;
        acc_y += w * obs.y;
        acc_z += w * obs.z;
        total_weight += w;
    }
    Quat pred;
    float inv = 1.0f / total_weight;
    pred.w = acc_w * inv;
    pred.x = acc_x * inv;
    pred.y = acc_y * inv;
    pred.z = acc_z * inv;
    return quat_normalize(pred);
}

int closure_check(Quat predicted, Quat observed) {
    return geodesic_dist(predicted, observed) > THRESHOLD;
}

static void genome_write(ClosureKernel *k, Quat context, Quat observed) {
    if (k->genome_count < GENOME_MAX) {
        k->genome[k->genome_count].context = context;
        k->genome[k->genome_count].observation = observed;
        k->genome_count++;
    } else {
        memmove(&k->genome[0], &k->genome[1],
                (GENOME_MAX - 1) * sizeof(GenomeEntry));
        k->genome[GENOME_MAX - 1].context = context;
        k->genome[GENOME_MAX - 1].observation = observed;
    }
}

static void init_operators(ClosureKernel *k) {
    const float phi = (1.0f + sqrtf(5.0f)) / 2.0f;
    const float two_pi = 6.28318530718f;
    const float pi = 3.14159265359f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        float theta1 = two_pi * (float)i / phi;
        float theta2 = acosf(1.0f - 2.0f * ((float)i + 0.5f) / (float)VOCAB_SIZE);
        float w = cosf(theta2/2.0f) * cosf(theta1/2.0f);
        float x = cosf(theta2/2.0f) * sinf(theta1/2.0f);
        float y = sinf(theta2/2.0f) * cosf(theta1/2.0f + pi*(float)i/(float)VOCAB_SIZE);
        float z = sinf(theta2/2.0f) * sinf(theta1/2.0f + pi*(float)i/(float)VOCAB_SIZE);
        Quat q = {w, x, y, z};
        k->operators[i] = quat_normalize(q);
    }
}

void kernel_init(ClosureKernel *k) {
    memset(k, 0, sizeof(ClosureKernel));
    init_operators(k);
    for (int c = 0; c < NUM_CHANNELS; c++) {
        k->cell_c[c].w = 1.0f;
    }
}

Quat process_token(ClosureKernel *k, uint8_t token) {
    if (token >= VOCAB_SIZE) token = VOCAB_SIZE - 1;

    for (int c = 0; c < NUM_CHANNELS; c++) {
        k->cell_c[c] = quat_mul(k->cell_c[c], k->operators[token]);
        k->cell_c[c] = quat_normalize(k->cell_c[c]);
    }

    Quat pred = knn_predict(k, k->cell_c[0]);
    Quat obs = k->operators[token];

    if (closure_check(pred, obs)) {
        genome_write(k, k->cell_c[0], obs);
        k->cell_c[0] = quat_slerp(k->cell_c[0], obs, SLERP_FAST);
    } else {
        k->cell_c[0] = quat_slerp(k->cell_c[0], obs, SLERP_SLOW);
    }

    k->cell_c[1] = quat_slerp(k->cell_c[1], obs, SLERP_SLOW / 5.0f);

    return pred;
}
