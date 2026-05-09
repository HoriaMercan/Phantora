#!/bin/bash
#SBATCH --job-name=phantora_64v
#SBATCH --output=logs/torchtitan_64v_%j.out
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=04:00:00

set -euo pipefail
ulimit -l unlimited
ulimit -s unlimited

SIF_IMAGE="/export/home/acs/stud/h/horia.mercan/licenta_build/phantora-original.sif"
WORKSPACE_DIR="$PWD"

# We simulate 64 Hosts, each with 1 GPU, all running on THIS single physical node
SIM_NODES="${SIM_NODES:-64}"
SIM_GPUS_PER_NODE="${SIM_GPUS_PER_NODE:-1}"
EVAL_VRAM_MIB="${EVAL_VRAM_MIB:-143771}"
EVAL_NGPU="${EVAL_NGPU:-1}" # Per virtual host

PHANTORA_CUSTOM_MODEL_PATH="${PHANTORA_CUSTOM_MODEL_PATH:-$WORKSPACE_DIR/custom_model_results.json}"
PHANTORA_BW_MBPS="${PHANTORA_BW_MBPS:-4500000.0}"
PHANTORA_LACKING_NODES="${PHANTORA_LACKING_NODES:-0.0}"
PHANTORA_DEFAULT_LATENCY_US="${PHANTORA_DEFAULT_LATENCY_US:-0.05}"
PHANTORA_CUSTOM_MODEL_TOPOLOGY="${PHANTORA_CUSTOM_MODEL_TOPOLOGY:-dragonfly}"

# 1) Build local fake /etc/hosts so virtual hosts can resolve each other and 'host-1' for rdzv
FAKE_HOSTS="/tmp/phantora_hosts_${SLURM_JOB_ID:-$$}"
echo "127.0.0.1 localhost" > "$FAKE_HOSTS"
for ((i=1; i<=SIM_NODES; i++)); do
    echo "127.0.0.1 host-$i" >> "$FAKE_HOSTS"
done

# 2) Generate the Netconfig (Apptainer execution without hostname yet)
apptainer exec --nv --bind "$WORKSPACE_DIR:/mnt" --pwd /mnt "$SIF_IMAGE" bash <<EOF
set -euo pipefail
export CARGO_HOME=/mnt/.cargo_home
mkdir -p "\$CARGO_HOME"
cd Phantora/phantora
cargo build --release

cd /mnt
python3 Phantora/tests/docker/torchtitan/config_gen.py \
  --nhost "$SIM_NODES" \
  --ngpu "$SIM_GPUS_PER_NODE" \
  # --custom_model "$PHANTORA_CUSTOM_MODEL_PATH" \
  # --bw_mbps "$PHANTORA_BW_MBPS" \
  # --lacking_nodes "$PHANTORA_LACKING_NODES" \
  # --default_latency_us "$PHANTORA_DEFAULT_LATENCY_US" \
  # --custom_model_topology "$PHANTORA_CUSTOM_MODEL_TOPOLOGY"

cd Phantora/phantora
# python3 build_graph.py
EOF

NETCONFIG_FILE="$WORKSPACE_DIR/Phantora/tests/docker/torchtitan/netconfig.toml"
PHANTORA_SOCKET_PREFIX="/tmp/phantora_${SLURM_JOB_ID:-$$}/phantora"
mkdir -p "$(dirname "$PHANTORA_SOCKET_PREFIX")"

# Cleanup function to kill background processes
cleanup() {
    echo "Cleaning up background tasks..."
    pkill -P $$ || true
    wait || true
    rm -f "$FAKE_HOSTS"
    rm -rf "$(dirname "$PHANTORA_SOCKET_PREFIX")"
}
trap cleanup EXIT

echo "Starting Simulator..."
touch "$WORKSPACE_DIR/visualizer-output" # Simulator needs this file to exist beforehand
apptainer exec --nv \
  --bind "$WORKSPACE_DIR:/mnt" \
  --bind "$FAKE_HOSTS:/etc/hosts" \
  --bind "$WORKSPACE_DIR/visualizer-output:/mnt/visualizer-output" \
  --pwd /mnt \
  "$SIF_IMAGE" bash -c "
    export PHANTORA_SOCKET_PREFIX=$PHANTORA_SOCKET_PREFIX
    export PHANTORA_LOG=\${PHANTORA_LOG:-info}
    ./Phantora/phantora/target/release/simulator --netconfig Phantora/tests/docker/torchtitan/netconfig.toml --timeline-file ./visualizer-output
" &
SIM_PID=$!

# Wait for the simulator socket
echo "Waiting for simulator socket..."
for _ in $(seq 1 30); do
  [[ -S "${PHANTORA_SOCKET_PREFIX}.simulator.sock" ]] && break
  sleep 1
done
if [[ ! -S "${PHANTORA_SOCKET_PREFIX}.simulator.sock" ]]; then
  echo "Failed to start simulator!"
  exit 1
fi
echo "Simulator is running."
# Introduce a small sleep so that the simulator fully binds
sleep 2

# 4) Start Host 1 in the background (node_rank 0) to establish rendezvous
# We start host-1 BEFORE the workers so the TCPStore spins up natively
echo "Starting host-1 (Master)..."
mkdir -p "$WORKSPACE_DIR/node-logs"
chmod 777 "$WORKSPACE_DIR/node-logs" # Ensure apptainer processes can write to it
torchrun_cmd="/phantora/dist/phantora_run torchrun --nproc_per_node $EVAL_NGPU --nnodes $SIM_NODES --rdzv_backend c10d --rdzv_endpoint=host-1:12345"
train_cmd="/phantora/tests/test_torchtitan.py --job.config_file=tests/test_torchtitan_llama3_8b.toml"

apptainer exec --nv \
  --bind "$WORKSPACE_DIR:/mnt" \
  --bind "$WORKSPACE_DIR/Phantora/tests:/phantora/tests" \
  --bind "$FAKE_HOSTS:/etc/hosts" \
  --hostname "host-1" \
  --pwd /phantora \
  "$SIF_IMAGE" bash -c "
    export NGPU=$EVAL_NGPU
    export PHANTORA_NGPU=$EVAL_NGPU
    export PHANTORA_VRAM_MIB=$EVAL_VRAM_MIB
    export PHANTORA_SOCKET_PREFIX=$PHANTORA_SOCKET_PREFIX
    export LD_PRELOAD=/phantora/dist/libcuda.so.1
    export LD_LIBRARY_PATH=/phantora/dist:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    $torchrun_cmd --node_rank 0 $train_cmd
" > "$WORKSPACE_DIR/node-logs/host-1.log" 2>&1 &
MASTER_PID=$!

# Wait for master to spin up TCPStore
echo "Waiting 60s for Master TCPStore to initialize..."
sleep 60

# 3) Start virtual hosts 2 through N in the background
echo "Starting $SIM_NODES virtual hosts..."
for ((w=2; w<=SIM_NODES; w++)); do
  apptainer exec --nv \
    --bind "$WORKSPACE_DIR:/mnt" \
    --bind "$WORKSPACE_DIR/Phantora/tests:/phantora/tests" \
    --bind "$FAKE_HOSTS:/etc/hosts" \
    --hostname "host-$w" \
    --pwd /phantora \
    "$SIF_IMAGE" bash -c "
      export NGPU=$EVAL_NGPU
      export PHANTORA_NGPU=$EVAL_NGPU
      export PHANTORA_VRAM_MIB=$EVAL_VRAM_MIB
      export PHANTORA_SOCKET_PREFIX=$PHANTORA_SOCKET_PREFIX
      export LD_PRELOAD=/phantora/dist/libcuda.so.1
      export LD_LIBRARY_PATH=/phantora/dist:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      # Use node_rank so torchrun knows which node it is natively
        $torchrun_cmd --node_rank $((w-1)) $train_cmd > /mnt/node-logs/host-${w}.log 2>&1
  " &
done

# Wait for Master specifically
wait $MASTER_PID
echo "Simulation complete! Host 1 finished."
