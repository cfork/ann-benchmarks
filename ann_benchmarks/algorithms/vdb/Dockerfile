FROM public.ecr.aws/n0k2r8k8/vdb:ann

RUN apt-get update
RUN apt-get install -y python3-numpy python3-scipy python3-pip build-essential git
RUN pip3 install -U pip
RUN python3 --version | grep 'Python 3.10'

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -r requirements.txt
RUN pip3 install "pinecone-client[grpc]"

RUN printf '#!/bin/bash\n\
/home/app/vdb >/dev/null 2>/dev/null &\n\
sleep 5\n\
python3 -u run_algorithm.py "$@"' > entrypoint.sh \
 && chmod u+x entrypoint.sh

ENTRYPOINT ["/home/app/entrypoint.sh"]
