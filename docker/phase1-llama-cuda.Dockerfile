# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_IMAGE
ARG CUDA_RUNTIME_IMAGE

FROM ${CUDA_DEVEL_IMAGE} AS build

ARG LLAMA_CPP_COMMIT
ARG CUDA_ARCHITECTURE=120

LABEL org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.revision="${LLAMA_CPP_COMMIT}"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        libgomp1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/llama.cpp
COPY . .

RUN test -f conversion/lfm2.py \
    && grep -q 'ModelBase.register("Lfm2ForCausalLM"' conversion/lfm2.py

RUN --mount=type=cache,id=lfm25-llama-cpp-build,target=/src/llama.cpp/build \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURE}" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_RPC=OFF \
        -DLLAMA_BUILD_COMMIT="${LLAMA_CPP_COMMIT}" \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_CURL=OFF \
    && cmake --build build --parallel \
    && cmake --install build --prefix /opt/llama-full \
    && mkdir -p /opt/llama-runtime/bin /opt/llama-runtime/lib \
    && cp \
        /opt/llama-full/bin/llama-bench \
        /opt/llama-full/bin/llama-cli \
        /opt/llama-full/bin/llama-gguf-hash \
        /opt/llama-full/bin/llama-imatrix \
        /opt/llama-full/bin/llama-perplexity \
        /opt/llama-full/bin/llama-quantize \
        /opt/llama-full/bin/llama-server \
        /opt/llama-full/bin/llama-tokenize \
        /opt/llama-runtime/bin/ \
    && cp -a /opt/llama-full/lib/*.so* /opt/llama-runtime/lib/

FROM ${CUDA_RUNTIME_IMAGE} AS toolchain

ARG LLAMA_CPP_COMMIT
ARG CUDA_ARCHITECTURE=120

LABEL org.opencontainers.image.title="Pinned llama.cpp CUDA toolchain for LFM2.5" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.revision="${LLAMA_CPP_COMMIT}" \
      ai.liquid.phase="1" \
      ai.liquid.cuda.architecture="${CUDA_ARCHITECTURE}"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/llama-runtime /opt/llama
COPY --from=build /src/llama.cpp/convert_hf_to_gguf.py /opt/llama-convert/
COPY --from=build /src/llama.cpp/conversion /opt/llama-convert/conversion
COPY --from=build /src/llama.cpp/gguf-py /opt/llama-convert/gguf-py
COPY --from=build /src/llama.cpp/requirements /opt/llama-convert/requirements

RUN python3 -m venv /opt/convert-venv \
    && /opt/convert-venv/bin/pip install --no-cache-dir \
        -r /opt/llama-convert/requirements/requirements-convert_hf_to_gguf.txt \
    && /opt/convert-venv/bin/pip check \
    && /opt/convert-venv/bin/pip freeze \
        > /opt/llama-convert/python-packages.lock.txt

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LD_LIBRARY_PATH="/opt/llama/lib" \
    PATH="/opt/llama/bin:/opt/convert-venv/bin:${PATH}"

WORKDIR /work
USER 65534:65534

CMD ["llama-cli", "--version"]
