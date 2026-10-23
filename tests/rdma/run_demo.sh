#!/bin/bash
#   ./run_demo.sh [nproj nz n]
#
set -euo pipefail

NPROJ=${1:-1024}
NZ=${2:-512}
N=${3:-2048}

SERVER=tomdet
CLIENT=tomo5
SERVER_IB=${SERVER_IB:-117.17.1.53}   # tomdet's IPoIB address (ibs2f1)
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN="source ~/miniforge3/etc/profile.d/conda.sh; conda activate tomocupy; cd $DIR"

# the server blocks forever if the client never shows up
trap 'ssh $SERVER "pkill -f tomo_server.py" 2>/dev/null || true' EXIT

echo "=== $SERVER: build the dataset in RAM and expose it to the HCA ==="
ssh $SERVER "$RUN; python -u tomo_server.py $NPROJ $NZ $N" &

# rdma.fetch retries while the server is still building, so no handshake here
echo "=== $CLIENT: pull it with RDMA READ and reconstruct ==="
ssh $CLIENT "$RUN; python -u tomo_client.py $SERVER_IB $NPROJ $NZ $N"
wait || true
