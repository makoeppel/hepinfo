FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
# install all needed system packages
RUN apt-get update && apt-get install -y build-essential git gfortran curl python-is-python3 python3-dev rsync wget cmake libfreetype6-dev nano zlib1g-dev

# install ROOT
RUN mkdir -p /opt && cd /opt \
  && wget https://root.cern/download/root_v6.20.04.Linux-ubuntu18-x86_64-gcc7.5.tar.gz \
  && tar xvf root_v6.20.04.Linux-ubuntu18-x86_64-gcc7.5.tar.gz \
  && rm root_v6.20.04.Linux-ubuntu18-x86_64-gcc7.5.tar.gz

ENV PATH=/opt/root/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/root/lib:$LD_LIBRARY_PATH
ENV ROOTSYS=/opt/root
ENV PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH

# install LHAPDF
RUN cd /opt && wget https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.5.5.tar.gz -O LHAPDF-6.5.5.tar.gz \
  && tar xf LHAPDF-6.5.5.tar.gz \
  && cd LHAPDF-6.5.5 && ./configure --prefix=/opt/lhapdf \
  && make && make install \
  && cd - \
  && rm -rf /opt/LHAPDF-6.5.5
ENV PATH=/opt/lhapdf/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/lhapdf/lib:$LD_LIBRARY_PATH

# install PDF sets
RUN wget http://lhapdfsets.web.cern.ch/lhapdfsets/current/CT10nlo.tar.gz -O- | tar xz -C /opt/lhapdf/share/LHAPDF \
 && wget http://lhapdfsets.web.cern.ch/lhapdfsets/current/CT10.tar.gz -O- | tar xz -C /opt/lhapdf/share/LHAPDF \
 && wget http://lhapdfsets.web.cern.ch/lhapdfsets/current/NNPDF23_lo_as_0130_qed.tar.gz -O- | tar xz -C /opt/lhapdf/share/LHAPDF

# install Pythia8
RUN curl -sL https://pythia.org/download/pythia83/pythia8312.tgz | tar -C /opt -zxf - \
  && cd /opt/pythia8312 \
  && ./configure --prefix=/opt/pythia8 --with-lhapdf6=/opt/lhapdf --with-gzip \
  && make install \
  && cd - \
  && rm -rf /opt/pythia8312
ENV PYTHIA8=/opt/pythia
ENV PATH=/opt/pythia8/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/pythia8/lib:$LD_LIBRARY_PATH

# install DELPHES
RUN cd /opt \
  && git clone https://github.com/delphes/delphes.git \
  && cd delphes && git checkout 3.5.0 \
  && mkdir build && cd build \
  && cmake -DCMAKE_INSTALL_PREFIX=/opt/delphes .. \
  && make install
ENV PATH=/opt/delphes/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/delphes/lib:$LD_LIBRARY_PATH


# install MadGraph
RUN cd /opt \
  && wget https://madgraph.mi.infn.it/Downloads/MG5_aMC_v2.9.2.tar.gz \
  && tar xfz MG5_aMC_v2.9.2.tar.gz && rm MG5_aMC_v2.9.2.tar.gz \
  && rm -f MG5_aMC_v2_9_2/input/.autoupdate
  
ENV PATH=/opt/MG5_aMC_v2_9_2/bin:$PATH

# create directory which should be mounted to the outside world
RUN mkdir /scratch && chmod 777 /scratch

ENV SHELL=/bin/bash
SHELL ["/bin/bash","-c"]
CMD ["/bin/bash"]
