FROM nvidia/cuda:10.0-cudnn7-runtime-centos7

CMD ["supervisord", "-c", "/opt/workspace/etc/supervisord.conf"]

RUN mkdir -p /opt/workspace /opt/data
WORKDIR /opt/workspace


RUN yum install -y epel-release gcc gcc-c++
RUN yum install -y python36-devel python36-pip nginx redis git
RUN pip3.6 install --upgrade pip

RUN pip3.6 install 'tensorflow-gpu==1.15.2' 'setuptools>=41.2'

COPY ./requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt

RUN cd /opt/ && git clone https://github.com/r9y9/wavenet_vocoder && cd /opt/wavenet_vocoder && git checkout v0.1.1
RUN cd /opt/wavenet_vocoder && pip3.6 install -e .[train]
RUN cd /opt/ && git clone https://github.com/r9y9/Tacotron-2 && cd /opt/Tacotron-2 && git checkout -B wavenet3 origin/wavenet3
RUN cd /opt/Tacotron-2 && pip3 install -r ./requirements.txt


COPY ./ /opt/workspace/
