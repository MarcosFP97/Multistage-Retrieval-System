FROM tensorflow/tensorflow:latest-gpu

WORKDIR /usr/src/app

RUN apt-get update &&  apt-get -y install python3
RUN apt-get -y install python3-pip 
RUN rm -f /etc/protocols \
    && apt-get -qqy install git \
    && git clone --recursive https://github.com/MarcosFP97/pygaggle.git \
    && pip3 install --upgrade pip \
    && pip3 install -r pygaggle/requirements.txt 
RUN apt-get install -y openjdk-11-jdk \
    && pip3 uninstall -y tensorflow \
    && pip3 install pandas \
    sentencepiece==0.1.91 \
    nltk \
    tqdm \
    tensorflow==2.3.1 \
    pyserini==0.9.4.0 \
    torch \
    bs4 \
    langdetect

COPY ./ /usr/src/app/pygaggle/

ENTRYPOINT ["python3", "pygaggle/cred_algorithm.py"]
