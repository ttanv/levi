#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

JAX_VERSION="0.4.38"
JAX_TARGET="${LEVI_BO_JAX_TARGET:-cpu}"

uv pip install --upgrade -r requirements.txt

case "$JAX_TARGET" in
  cpu)
    uv pip install --upgrade "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
    ;;
  cuda12)
    uv pip install --upgrade \
      --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
      "jax[cuda12]==${JAX_VERSION}"
    ;;
  *)
    echo "Unsupported LEVI_BO_JAX_TARGET: ${JAX_TARGET}" >&2
    echo "Use LEVI_BO_JAX_TARGET=cpu or LEVI_BO_JAX_TARGET=cuda12." >&2
    exit 1
    ;;
esac
