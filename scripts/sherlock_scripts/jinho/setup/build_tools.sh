# Load modules
module load gcc/12.4.0

# Define installation path
SOFTWARE_PATH=/oak/stanford/groups/possu/zhkim216/software
# SOFTWARE_PATH=/media/scratch/software
mkdir -p "${SOFTWARE_PATH}"
cd "${SOFTWARE_PATH}"

# Install foldseek
if [ ! -d "${SOFTWARE_PATH}/foldseek" ]; then
    wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
    tar -xzvf foldseek-linux-avx2.tar.gz
fi


# Install mmseqs2
if [ ! -d "${SOFTWARE_PATH}/mmseqs" ]; then
    wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
    tar -xzvf mmseqs-linux-avx2.tar.gz
fi

# Build Redis
# check if Redis is already installed
if [ ! -d "${SOFTWARE_PATH}/redis" ]; then
    if [ ! -f redis-stable.tar.gz ]; then
        wget https://download.redis.io/redis-stable.tar.gz
    fi
    tar -xzvf redis-stable.tar.gz
    cd redis-stable
    make
    make install PREFIX="${SOFTWARE_PATH}/redis"
    cd ..
fi

# Build master-v1.6
# check if master-v1.6 is already installed
if [ ! -d "${SOFTWARE_PATH}/master-v1.6" ]; then
    if [ ! -f master-v1.6.tar.gz ]; then
        echo "master-v1.6.tar.gz not found. Please place it in ${SOFTWARE_PATH} or update this script with the correct URL."
        exit 1
    fi
    tar -xzvf master-v1.6.tar.gz
    cd master-v1.6

    # patch out static linking (due to "-lm" or "-lc" errors)
    sed -i.bak '/^ifeq.*Linux/,/^endif/d' makefile

    # now build
    make all
    cd ..
fi
