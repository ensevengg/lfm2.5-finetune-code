FROM python@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ARG EVALPLUS_REVISION=26d6d00bb1fd0fa37f39c99d5290da67891d1c5e
ARG EVALPLUS_PACKAGE_VERSION=0.3.1.post44

LABEL org.opencontainers.image.title="LFM2.5 EvalPlus E2 scorer" \
      org.opencontainers.image.source="https://github.com/evalplus/evalplus" \
      org.opencontainers.image.revision="${EVALPLUS_REVISION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY . /opt/evalplus

RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EVALPLUS=${EVALPLUS_PACKAGE_VERSION} \
        python -m pip install /opt/evalplus \
    && python -m pip check \
    && python -c "import evalplus.evaluate; print('EvalPlus scorer import: PASS')"

WORKDIR /work

CMD ["python", "-m", "evalplus.evaluate", "--help"]
