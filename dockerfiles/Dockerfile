ARG from=ubuntu:focal
FROM ${from}

SHELL ["/bin/bash", "-i", "-c"]

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        sudo \
        gosu &&\
    rm -rf /var/lib/apt/lists/*

# Setup locales
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&\
    apt-get install -y --no-install-recommends locales locales-all &&\
    rm -rf /var/lib/apt/lists/*

# Utilities
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        software-properties-common \
        apt-transport-https \
        apt-utils \
        wget \
        nano \
        dbus-x11 \
        tree \
        bash-completion \
        gnupg2 \
        colordiff \
        trash-cli \
        xterm \
        curl \
        zip \
        unzip \
        mesa-utils \
        &&\
    rm -rf /var/lib/apt/lists/*

# Build and development tools
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        cmake-curses-gui \
        ninja-build \
        gcc-8 \
        g++-8 \
        doxygen \
        swig \
        &&\
    rm -rf /var/lib/apt/lists/*

# Install and setup python tools
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3-dev \
        python3-wheel \
        python3-pygments \
        python3-tk \
        &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf $HOME/.cache/pip &&\
    echo "alias python=python3" >> /etc/bash.bashrc

# Install Gazebo
RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" \
        > /etc/apt/sources.list.d/gazebo-stable.list &&\
    wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - &&\
    apt-get update &&\
    apt-get install --no-install-recommends -y \
        gazebo11 \
        libgazebo11-dev \
        &&\
    rm -rf /var/lib/apt/lists/* &&\
    echo "[[ -f /usr/share/gazebo/setup.sh ]] && source /usr/share/gazebo/setup.sh" >> /etc/bash.bashrc

# Install libraries and tools
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        python3-vcstools \
        python3-numpy \
        libeigen3-dev \
        coinor-libipopt-dev \
        gfortran \
        libboost-system-dev \
        libboost-filesystem-dev \
        libboost-thread-dev \
        libirrlicht-dev \
        libtinyxml-dev \
        libedit-dev \
        libace-dev \
        libgsl0-dev \
        libopencv-dev \
        libode-dev \
        liblua5.1-dev \
        lua5.1 \
        qtbase5-dev \
        qtdeclarative5-dev \
        qtmultimedia5-dev \
        qml-module-qtquick2 \
        qml-module-qtquick-window2 \
        qml-module-qtmultimedia \
        qml-module-qtquick-dialogs \
        qml-module-qtquick-controls \
        qml-module-qt-labs-folderlistmodel \
        qml-module-qt-labs-settings \
        libsdl1.2-dev \
        libxml2-dev \
        libv4l-dev \
        libspdlog-dev \
        nlohmann-json3-dev \
        libmatio-dev \
        libqt5opengl5-dev \
        &&\
    rm -rf /var/lib/apt/lists/*

# Build configuration
ENV CMAKE_GENERATOR="Ninja"
ENV SOURCES_BUILD_TYPE="Release"

# Install pybind11
RUN git clone https://github.com/pybind/pybind11.git &&\
    cd pybind11 &&\
    git checkout bca4b36ba75957b463463a35ad7dc98b8326b04c &&\
    mkdir build && cd build &&\
    cmake .. &&\
    sudo ninja install

# Environment setup for the robotology-superbuild repositories
ENV IIT_DIR=/iit
ENV IIT_INSTALL=${IIT_DIR}/local
ENV IIT_SOURCES=${IIT_DIR}/sources
ARG IIT_BIN=${IIT_INSTALL}/bin

# Download robotology-superbuild sources
RUN mkdir -p ${IIT_SOURCES} &&\
    cd ${IIT_SOURCES} &&\
    git clone https://github.com/robotology/robotology-superbuild.git

# Build robotology-superbuild sources
EXPOSE 10000/tcp
ENV YARP_COLORED_OUTPUT=1
ENV YARP_ROBOT_NAME="iCubGazeboV2_5"
ARG ROBOTOLOGY_PROJECT_TAGS=Custom
COPY robotology.yaml ${IIT_SOURCES}/robotology-superbuild
RUN cd ${IIT_SOURCES}/robotology-superbuild &&\
    git checkout 55f77177a9b28947048d0601d03fbd5b1eee3680 &&\
    mkdir -p build && cd build &&\
    export CC=gcc && export CXX=g++ &&\
    cmake \
          -GNinja \
          -DCMAKE_BUILD_TYPE=Release \
          -DROBOTOLOGY_ENABLE_CORE:BOOL=ON \
          -DROBOTOLOGY_ENABLE_DYNAMICS:BOOL=ON \
          -DROBOTOLOGY_ENABLE_DYNAMICS_FULL_DEPS:BOOL=ON \
          -DROBOTOLOGY_ENABLE_ROBOT_TESTING:BOOL=OFF \
          -DROBOTOLOGY_ENABLE_HUMAN_DYNAMICS:BOOL=ON \
          -DROBOTOLOGY_USES_GAZEBO:BOOL=ON \
          -DROBOTOLOGY_USES_PYTHON:BOOL=ON \
          -DROBOTOLOGY_USES_OCTAVE:BOOL=OFF \
          -DROBOTOLOGY_USES_MATLAB:BOOL=OFF \
          -DNON_INTERACTIVE_BUILD:BOOL=ON \
          -DBUILD_TESTING:BOOL=OFF \
          -DYCM_DISABLE_SYSTEM_PACKAGES:BOOL=ON \
          -DROBOTOLOGY_PROJECT_TAGS=${ROBOTOLOGY_PROJECT_TAGS} \
          -DROBOTOLOGY_PROJECT_TAGS_CUSTOM_FILE=${IIT_SOURCES}/robotology-superbuild/robotology.yaml \
          -DYCM_EP_ADDITIONAL_CMAKE_ARGS='-DFRAMEWORK_COMPILE_PYTHON_BINDINGS:BOOL=ON -DPython3_ROOT_DIR=$(python3 -c "import sys; print(sys.prefix)") -DFRAMEWORK_USE_Python3:BOOL=ON -DFRAMEWORK_USE_pybind11:BOOL=ON' \
          .. &&\
    ninja &&\
    ln -s ${IIT_SOURCES}/robotology-superbuild/build/install ${IIT_INSTALL} &&\
    echo "source ${IIT_INSTALL}/share/robotology-superbuild/setup.sh" >> /etc/bash.bashrc &&\
    ln -s ${IIT_SOURCES}/robotology-superbuild/src/YARP/scripts/yarp_completion /etc/bash_completion.d/yarp_completion &&\
    find src/ -type f -not -name 'CMakeCache.txt' -delete

# Environment setup for Ignition
ENV IGNITION_DIR=/ignition
ENV IGNITION_INSTALL=${IGNITION_DIR}/local
ENV IGNITION_SOURCES=${IGNITION_DIR}/sources
ENV PATH=${IGNITION_INSTALL}/bin:${PATH}
ENV CMAKE_PREFIX_PATH=${IGNITION_INSTALL}:${CMAKE_PREFIX_PATH}

# Build Ignition from sources
ARG IGNITION_DEFAULT_CHANNEL="stable"
RUN mkdir -p ${IGNITION_SOURCES}
COPY ignition.yaml ${IGNITION_SOURCES}/workspace/src/ignition.yaml
RUN pip install vcstool colcon-common-extensions &&\
    rm -r $HOME/.cache/pip &&\
    echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-${IGNITION_DEFAULT_CHANNEL} `lsb_release -cs` main" > \
        /etc/apt/sources.list.d/gazebo-${IGNITION_DEFAULT_CHANNEL}.list &&\
    wget http://packages.osrfoundation.org/gazebo.key -qO - | apt-key add - &&\
    cd ${IGNITION_SOURCES}/workspace/src &&\
    vcs import < ignition.yaml &&\
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4 &&\
    apt-get update &&\
    sudo apt -y install --no-install-recommends \
        $(sort -u $(find . -iname 'packages-'$(lsb_release -cs)'.apt' -o -iname 'packages.apt') | grep -v -E "dart|^libignition|^libsdformat" | tr '\n' ' ') &&\
    rm -rf /var/lib/apt/lists/* &&\
    cd ${IGNITION_SOURCES}/workspace &&\
    colcon graph &&\
    export CC=gcc && export CXX=g++ &&\
    colcon build \
        --cmake-args \
        -G"$CMAKE_GENERATOR" \
        -DBUILD_TESTING:BOOL=OFF \
        -DCMAKE_BUILD_TYPE=${SOURCES_BUILD_TYPE} \
        --merge-install \
        &&\
    find build/ -type f -not -name 'CMakeCache.txt' -delete &&\
    echo "[[ -f ${IGNITION_SOURCES}/workspace/install/setup.bash ]] && source ${IGNITION_SOURCES}/workspace/install/setup.bash" >> /etc/bash.bashrc

# Install Adherent
RUN pip install setuptools_scm==6.4.2 &&\
    rm -r $HOME/.cache/pip &&\
    git clone https://github.com/ami-iit/paper_viceconte_2021_ral_adherent.git adherent &&\
    cd adherent &&\
    pip install -e .

# Download Adherent dataset
RUN cd /adherent &&\
    rm -rf datasets &&\
    curl -O https://zenodo.org/record/6201915/files/datasets.zip