# llm-scaler v28dbg — flight-recorder debug overlay on adv:v27 (KNOW_ISSUES
# #11 graphs-x-spec residual). NOT a serving image: fr.log writes one line
# per replay/collective and the markers are for wedge localization only.
FROM llm-scaler-vllm-adv:v27

COPY fr.py /opt/venv/lib/python3.12/site-packages/fr.py
COPY cudagraph_utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/cudagraph_utils.py
COPY xpu_communicator.py /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q "llm-scaler v28dbg" $SP/vllm/v1/worker/gpu/cudagraph_utils.py && \
    grep -q "llm-scaler v28dbg" $SP/vllm/distributed/device_communicators/xpu_communicator.py && \
    grep -q "VLLM_XPU_FR" $SP/fr.py && \
    /opt/venv/bin/python -m py_compile \
      $SP/fr.py \
      $SP/vllm/v1/worker/gpu/cudagraph_utils.py \
      $SP/vllm/distributed/device_communicators/xpu_communicator.py && \
    find $SP -name '__pycache__' -path '*v28dbg*' -prune -o -name '*.pyc' -newer $SP/fr.py -print | grep -v v28dbg | xargs -r rm -f && \
    echo "v28dbg overlay OK"
