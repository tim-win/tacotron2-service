FROM nvidia/cuda:10.0-cudnn7-runtime-centos8

CMD ["supervisord", "-c", "/opt/workspace/etc/supervisord.conf"]

RUN mkdir -p /opt/workspace /opt/data
WORKDIR /opt/workspace


RUN yum install -y epel-release gcc
RUN yum install -y python36-devel python36-pip nginx redis git
RUN pip3.6 install --upgrade pip

RUN pip3.6 install 'tensorflow-gpu>=2,<3'

COPY ./requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt

RUN cd /opt/ && git clone https://github.com/r9y9/wavenet_vocoder
RUN cd /opt/ && git clone https://github.com/r9y9/Tacotron-2

RUN cd /opt/wavenet_vocoder && pip3.6 install -e .

COPY ./ /opt/workspace/

