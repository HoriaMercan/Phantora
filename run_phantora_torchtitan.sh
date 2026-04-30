#!/bin/bash
#SBATCH --job-name=phantora_torchtitan
#SBATCH --output=logs/torchtitan_%j.out
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
ulimit -l unlimited
ulimit -s unlimited
SIF_IMAGE="/export/home/acs/stud/h/horia.mercan/licenta_build/phantora-original.sif"
WORKSPACE_DIR="$PWD"
SIM_NODES="${SIM_NODES:-8}"
SIM_GPUS_PER_NODE="${SIM_GPUS_PER_NODE:-8}"
EVAL_NNODES="${EVAL_NNODES:-1}"
EVAL_NGPU="${EVAL_NGPU:-3}"
EVAL_VRAM_MIB="${EVAL_VRAM_MIB:-81920}"
PHANTORA_CUSTOM_MODEL_PATH="${PHANTORA_CUSTOM_MODEL_PATH:-$WORKSPACE_DIR/custom_model_results.json}"
PHANTORA_BW_MBPS="${PHANTORA_BW_MBPS:-450000.0}"
PHANTORA_LACKING_NODES="${PHANTORA_LACKING_NODES:-0.0}"
PHANTORA_DEFAULT_LATENCY_US="${PHANTORA_DEFAULT_LATENCY_US:-1.0}"
PHANTORA_CUSTOM_MODEL_TOPOLOGY="${PHANTORA_CUSTOM_MODEL_TOPOLOGY:-fattree}"

export SIM_NODES SIM_GPUS_PER_NODE EVAL_NNODES EVAL_NGPU EVAL_VRAM_MIB
export PHANTORA_CUSTOM_MODEL_PATH PHANTORA_BW_MBPS PHANTORA_LACKING_NODES PHANTORA_DEFAULT_LATENCY_US PHANTORA_CUSTOM_MODEL_TOPOLOGY

RUNTIME_HOSTFILE="$WORKSPACE_DIR/tests/docker/torchtitan/hostfile.runtime"
mkdir -p "$(dirname "$RUNTIME_HOSTFILE")"

if [[ -z "${SLURM_NODELIST:-}" ]]; then
        printf '%s slots=%s\n' "$(hostname)" "$EVAL_NGPU" > "$RUNTIME_HOSTFILE"
else
        mapfile -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
        if (( ${#SLURM_HOSTS[@]} < EVAL_NNODES )); then
                echo "ERROR: Requested EVAL_NNODES=$EVAL_NNODES but only ${#SLURM_HOSTS[@]} host(s) in SLURM_NODELIST."
                exit 1
        fi

        srun apptainer exec --nv --bind "$WORKSPACE_DIR:/mnt" --pwd /mnt "$SIF_IMAGE" bash <<'EOF'
        set -euo pipefail

        REPO_ROOT=/mnt
        [[ -d /mnt/Phantora ]] && REPO_ROOT=/mnt/Phantora

        export CUDA_HOME=/usr/local/cuda
        export CARGO_HOME=/mnt/.cargo_home
        export LD_LIBRARY_PATH=/phantora/dist:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
        mkdir -p "$CARGO_HOME"

        cleanup() {
          if [[ -n "${SIM_PID:-}" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
            kill "$SIM_PID" || true
            wait "$SIM_PID" || true
          fi
        }
        trap cleanup EXIT

        cd "$REPO_ROOT/phantora"
        cargo build --release

        cd "$REPO_ROOT"
        python3 tests/docker/torchtitan/config_gen.py \
          --nhost "$SIM_NODES" \
          --ngpu "$SIM_GPUS_PER_NODE" \
          --custom_model "$PHANTORA_CUSTOM_MODEL_PATH" \
          --bw_mbps "$PHANTORA_BW_MBPS" \
          --lacking_nodes "$PHANTORA_LACKING_NODES" \
          --default_latency_us "$PHANTORA_DEFAULT_LATENCY_US" \
          --custom_model_topology "$PHANTORA_CUSTOM_MODEL_TOPOLOGY"

        NETCONFIG_FILE="$REPO_ROOT/tests/docker/torchtitan/netconfig.toml"
        HOSTFILE="$REPO_ROOT/tests/docker/torchtitan/hostfile.runtime"
        mapfile -t RUNTIME_HOSTS < <(awk '{print $1}' "$HOSTFILE")
        for ((i = 1; i <= SIM_NODES; i++)); do
          sed -i "s/host-$i/${RUNTIME_HOSTS[$(((i - 1) % ${#RUNTIME_HOSTS[@]}))]}/g" "$NETCONFIG_FILE"
        done

        cd "$REPO_ROOT/phantora"
        python3 build_graph.py
        PHANTORA_SOCKET_PREFIX="/tmp/phantora_${SLURM_JOB_ID:-$$}"
        ./target/release/simulator --netconfig "$NETCONFIG_FILE" &
        SIM_PID=$!

        for _ in $(seq 1 30); do
          [[ -S "${PHANTORA_SOCKET_PREFIX}.simulator.sock" ]] && break
          sleep 1
        done
        [[ -S "${PHANTORA_SOCKET_PREFIX}.simulator.sock" ]]

        cd "$REPO_ROOT"
        PHANTORA_SOCKET_PREFIX="$PHANTORA_SOCKET_PREFIX" \
        PHANTORA_VRAM_MIB="$EVAL_VRAM_MIB" \
        PHANTORA_NGPU="$EVAL_NGPU" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        LD_PRELOAD=/phantora/dist/libcuda.so.1 \
        /phantora/dist/phantora_run torchrun \
          --nproc_per_node "$EVAL_NGPU" \
          --nnodes "$EVAL_NNODES" \
          tests/test_torchtitan.py \
          --job.config_file=tests/test_torchtitan_llama3_8b.toml
        EOF

        echo "Job finished."
        # TorchTitan is finished, tell the background simulator to shut down
        kill "\$SIM_PID" || true
        wait "\$SIM_PID" || true
        echo 'Simulation complete.'
    "

echo "Job finished."
