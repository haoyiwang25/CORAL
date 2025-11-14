FROM  nvcr.io/nvidia/pytorch:22.09-py3

COPY ./ /workspace/coral
RUN pip install -r /workspace/coral/requirements.txt
