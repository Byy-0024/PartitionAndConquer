EXE=../bin/main
TASK=partition
GRAPHDIR=../dataset/bio-human-gene1
PARTITION_METHODS=(
    # URP \
    GRP \
    # FGP
)
PARTITION_NUMS=(
    4 \
    8 \
    16 \
    32
)

for method in "${PARTITION_METHODS[@]}"; do
    OUTPUTDIR=${GRAPHDIR}/partition/${method}
    mkdir -p ${OUTPUTDIR}
    for num in "${PARTITION_NUMS[@]}"; do
        echo "Running $TASK on $GRAPHDIR with method $method and num $num"
        ${EXE} ${TASK} ${GRAPHDIR} ${num} ${method}
    done
done