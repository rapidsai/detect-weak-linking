/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shared_kernel.cuh"

__global__ void kernel_only_in_a(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f;
}

void launch_a(float* out, const float* a, const float* b, int n) {
    shared_add<<<1, 256>>>(out, a, b, n);
    kernel_only_in_a<<<1, 256>>>(out, n);
}
