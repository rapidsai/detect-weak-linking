#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# default env is called `rapids` and will be used without specifying `-n`
# we _could_ be more explicit here but that name might bitrot, so we rely on
# "install into the current active env" behavior
conda install -y -c conda-forge \
    cuda-nvcc cuda-cuobjdump cuda-cudart-dev \
    cxx-compiler binutils \
    pytest

pytest tests/ -v
