#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

conda install -y -n base -c conda-forge \
    cuda-nvcc cuda-cuobjdump cuda-cudart-dev \
    cxx-compiler binutils \
    pytest

pytest tests/ -v
