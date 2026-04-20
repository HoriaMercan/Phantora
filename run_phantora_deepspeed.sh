#!/bin/bash
#SBATCH --job-name=phantora_deepspeed
#SBATCH --output=logs/deepspeed_%j.out
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail

# --- ENVIRONMENT SETTINGS ---
# Optional: Load CUDA module if required by your host/cluster setup
# module load cuda

# Prevent memory lock issues for deep learning tasks
ulimit -l unlimited
ulimit -s unlimited

# --- CONFIGURATION ---
SIF_IMAGE="/export/home/acs/stud/h/horia.mercan/licenta_build/phantora-original.sif"
WORKSPACE_DIR="$PWD"
SLURM_NNODES="${SLURM_NNODES:-1}"
SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-2}"

# Runtime parameters (DeepSpeed world size) - derive from SLURM by default
EVAL_NNODES="${EVAL_NNODES:-$SLURM_NNODES}"
EVAL_NGPU="${EVAL_NGPU:-$SLURM_GPUS_PER_NODE}"
EVAL_VRAM_MIB="${EVAL_VRAM_MIB:-81920}"

# Simulation parameters - default to runtime world shape
SIM_NODES="${SIM_NODES:-$EVAL_NNODES}"
SIM_GPUS_PER_NODE="${SIM_GPUS_PER_NODE:-$EVAL_NGPU}"


PHANTORA_CUSTOM_MODEL_PATH="${PHANTORA_CUSTOM_MODEL_PATH:-$WORKSPACE_DIR/custom_model_results.json}"
PHANTORA_BW_MBPS="${PHANTORA_BW_MBPS:-3600000.0}"
PHANTORA_LACKING_NODES="${PHANTORA_LACKING_NODES:-0.0}"
PHANTORA_DEFAULT_LATENCY_US="${PHANTORA_DEFAULT_LATENCY_US:-1.0}"
PHANTORA_CUSTOM_MODEL_TOPOLOGY="${PHANTORA_CUSTOM_MODEL_TOPOLOGY:-dragonfly}"

if [[ "$SIM_NODES" != "$EVAL_NNODES" ]]; then
        echo "INFO: SIM_NODES ($SIM_NODES) differs from EVAL_NNODES ($EVAL_NNODES)."
        echo "      Simulated hosts will be mapped round-robin onto available runtime hostnames."
fi

RUNTIME_HOSTFILE="$WORKSPACE_DIR/tests/docker/deepspeed/hostfile.runtime"
mkdir -p "$(dirname "$RUNTIME_HOSTFILE")"

if [[ "$EVAL_NNODES" == "1" ]]; then
        THIS_HOST=$(hostname)
        printf '%s slots=%s\n' "$THIS_HOST" "$EVAL_NGPU" > "$RUNTIME_HOSTFILE"
else
        if [[ -z "${SLURM_NODELIST:-}" ]]; then
                echo "ERROR: SLURM_NODELIST is empty for multi-node run."
                exit 1
        fi

        mapfile -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
        if (( ${#SLURM_HOSTS[@]} < EVAL_NNODES )); then
                echo "ERROR: Requested EVAL_NNODES=$EVAL_NNODES but only ${#SLURM_HOSTS[@]} host(s) in SLURM_NODELIST."
                exit 1
        fi

        : > "$RUNTIME_HOSTFILE"
        for ((i = 0; i < EVAL_NNODES; i++)); do
                printf '%s slots=%s\n' "${SLURM_HOSTS[$i]}" "$EVAL_NGPU" >> "$RUNTIME_HOSTFILE"
        done
fi

echo "DeepSpeed runtime hostfile: $RUNTIME_HOSTFILE"
cat "$RUNTIME_HOSTFILE"
echo "----------------------------------------------------"
echo "Running Phantora DeepSpeed Test Job"
echo "Image: $SIF_IMAGE"
echo "Workspace: $WORKSPACE_DIR"
echo "Simulating: $SIM_NODES nodes × $SIM_GPUS_PER_NODE GPUs/node"
echo "Evaluating: $EVAL_NNODES node(s) × $EVAL_NGPU GPU(s)/node"
echo "----------------------------------------------------"

# Make sure the logs directory exists
mkdir -p logs

# --- EXECUTION ---
# Using srun to allocate the resources within the SLURM allocation.
# We bind the current working directory to /mnt. 
# The commands are passed via bash -c to run sequentially within the container environment.

srun apptainer exec --nv \
    --bind "$WORKSPACE_DIR:/mnt" \
    --pwd /mnt \
    "$SIF_IMAGE" bash -c "
                set -euo pipefail

        # 1. Setup paths
        export CUDA_HOME=/usr/local/cuda
        export TORCH_LIB=\$(python3 -c \"import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))\")
        export PYTHON_LIB=\$(python3 -c \"import sysconfig; print(sysconfig.get_config_var('LIBDIR'))\")
        
                # Build library search path with real CUDA runtime first.
                # Accept CUDA 12 or CUDA 11 runtimes depending on how flash-attn was built.
                REAL_CUDART=\$(ldconfig -p 2>/dev/null | awk '/libcudart.so.12/ {print \$NF; exit}')
                if [[ -z \"\$REAL_CUDART\" ]]; then
                        REAL_CUDART=\$(ldconfig -p 2>/dev/null | awk '/libcudart.so.11/ {print \$NF; exit}')
                fi
                if [[ -z \"\$REAL_CUDART\" ]]; then
                        REAL_CUDART=\$(ldconfig -p 2>/dev/null | awk '/libcudart.so\$/ {print \$NF; exit}')
                fi
                if [[ -z \"\$REAL_CUDART\" ]]; then
                        for candidate in \\
                                /.singularity.d/libs/libcudart.so.12 \\
                                /.singularity.d/libs/libcudart.so.11 \\
                                /.singularity.d/libs/libcudart.so \\
                                /usr/lib64/libcudart.so.12 \\
                                /usr/lib64/libcudart.so.11 \\
                                /usr/lib64/libcudart.so \\
                                /usr/lib/x86_64-linux-gnu/libcudart.so.12 \\
                                /usr/lib/x86_64-linux-gnu/libcudart.so.11 \\
                                /usr/lib/x86_64-linux-gnu/libcudart.so \\
                                \$CUDA_HOME/lib64/libcudart.so.12 \\
                                \$CUDA_HOME/lib64/libcudart.so.11 \\
                                \$CUDA_HOME/lib64/libcudart.so; do
                                if [[ -e \"\$candidate\" ]]; then
                                        REAL_CUDART=\"\$candidate\"
                                        break
                                fi
                        done
                fi

                if [[ -z \"\$REAL_CUDART\" ]]; then
                        echo 'ERROR: libcudart not found in container runtime.'
                        exit 1
                fi

                REAL_CUDART_DIR=\$(dirname \"\$REAL_CUDART\")
                BASE_LD_LIBRARY_PATH=\$REAL_CUDART_DIR:\$TORCH_LIB:\$PYTHON_LIB:\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
                export LD_LIBRARY_PATH=\$BASE_LD_LIBRARY_PATH
                export LIBRARY_PATH=\$LD_LIBRARY_PATH

        # 2. Configure Cargo
        export CARGO_HOME=/mnt/.cargo_home
        mkdir -p \$CARGO_HOME

        export SOCKET_DIR=\"/tmp/phantora_\$(date +%s)_$SLURM_JOB_ID\"
        mkdir -p \$SOCKET_DIR
        chmod 777 \$SOCKET_DIR
        export PHANTORA_SOCKET_PREFIX=\"\$SOCKET_DIR\"

        cleanup() {
            if [[ -n \"\${SIM_PID:-}\" ]] && kill -0 \"\$SIM_PID\" 2>/dev/null; then
                kill \"\$SIM_PID\" || true
                wait \"\$SIM_PID\" || true
            fi
            rm -rf \"\$SOCKET_DIR\" || true
        }
        trap cleanup EXIT

                # 3. Build simulator binary
        cd /mnt/Phantora/phantora
        cargo build --release || { echo 'Cargo build failed'; exit 1; }

        # 4. Prepare args
                EVAL_NNODES=$EVAL_NNODES
                EVAL_NGPU=$EVAL_NGPU
                EVAL_VRAM_MIB=$EVAL_VRAM_MIB
                SIM_NODES=$SIM_NODES
                SIM_GPUS_PER_NODE=$SIM_GPUS_PER_NODE

                # 5. Generate config files from deepspeed template files
                cd /mnt/Phantora
                echo 'Generating netconfig.toml, hostfile, and deepspeed_env with config_gen.py...'
                python3 tests/docker/deepspeed/config_gen.py \\
                        --nhost $SIM_NODES \\
                        --ngpu $SIM_GPUS_PER_NODE \
                        --custom_model \"${PHANTORA_CUSTOM_MODEL_PATH:-}\" \
                        --bw_mbps $PHANTORA_BW_MBPS \
                        --lacking_nodes $PHANTORA_LACKING_NODES \
                        --default_latency_us $PHANTORA_DEFAULT_LATENCY_US \
                        --custom_model_topology $PHANTORA_CUSTOM_MODEL_TOPOLOGY

                NETCONFIG_FILE=/mnt/Phantora/tests/docker/deepspeed/netconfig.toml
                HOSTFILE=/mnt/Phantora/tests/docker/deepspeed/hostfile.runtime
                DEEPSPEED_ENV=/mnt/Phantora/tests/docker/deepspeed/deepspeed_env

                # Replace logical host-i entries in netconfig with runtime hostnames.
                # If SIM_NODES > runtime hosts, distribute simulated hosts round-robin.
                mapfile -t RUNTIME_HOSTS < <(awk '{print \$1}' "\$HOSTFILE")
                if [[ "\${#RUNTIME_HOSTS[@]}" -eq 0 ]]; then
                        echo "ERROR: runtime hostfile has no hosts: \$HOSTFILE"
                        exit 1
                fi

                for ((i = 1; i <= SIM_NODES; i++)); do
                        host_index=\$(((i - 1) % \${#RUNTIME_HOSTS[@]}))
                        runtime_host=\${RUNTIME_HOSTS[\$host_index]}
                        sed -i \"s/host-\$i/\$runtime_host/g\" \"\$NETCONFIG_FILE\"
                done
        
                # 6. Start Simulator
                cd /mnt/Phantora/phantora
        echo 'Building computational graph...'
        python3 build_graph.py

        echo 'Starting Phantora Simulator server...'
                PHANTORA_SOCKET_PREFIX=\$PHANTORA_SOCKET_PREFIX \\
                LD_PRELOAD= \\
                LD_LIBRARY_PATH=\$BASE_LD_LIBRARY_PATH \\
                RUST_BACKTRACE=full ./target/release/simulator \\
                        --netconfig \"\$NETCONFIG_FILE\" &
        
        SIM_PID=\$!

                for _ in \$(seq 1 30); do
                        if [[ -S \"\${PHANTORA_SOCKET_PREFIX}.simulator.sock\" ]]; then
                                break
                        fi
                        sleep 1
                done

                if [[ ! -S \"\${PHANTORA_SOCKET_PREFIX}.simulator.sock\" ]]; then
                        echo 'Simulator socket was not created in time. Exiting.'
                        exit 1
                fi
        
                # 7. Run DeepSpeed
        echo 'Starting DeepSpeed training simulation...'
        cd /mnt/Phantora

        # For single-node simulations, use --include instead of hostfile (no SSH needed)
        # For multi-node, use the hostfile
        DEEPSPEED_LAUNCHER_ARGS=""
        if [[ \"\$EVAL_NNODES\" == \"1\" ]]; then
                DEEPSPEED_LAUNCHER_ARGS="--num_gpus \$EVAL_NGPU"
        else
                DEEPSPEED_LAUNCHER_ARGS=\"-H \$HOSTFILE\"
        fi

        # FORCE Python to use the library that WE KNOW has the symbol
        # We point LD_PRELOAD to the one inside the container (/phantora/...)
        PHANTORA_SOCKET_PREFIX=\$PHANTORA_SOCKET_PREFIX \\
                PHANTORA_VRAM_MIB=\$EVAL_VRAM_MIB \\
                PHANTORA_NGPU=\$EVAL_NGPU \\
                PHANTORA_IGNORE_CPU_TIME=1 \\
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
                LD_LIBRARY_PATH=/phantora/dist:\$BASE_LD_LIBRARY_PATH \\
                LD_PRELOAD=/phantora/dist/libcuda.so.1 \\
                /phantora/dist/phantora_run deepspeed \\
                \$DEEPSPEED_LAUNCHER_ARGS \\
                tests/test_deepspeed.py \\
                --num_layers 12 \\
                --hidden_size 1024 \\
                --ffn_hidden_size 2816 \\
                --num_attention_heads 8 \\
                --vocab_size 32000 \\
                --sequence_length 512 \\
                --micro_batch_size 1 \\
                --iterations 4

        # DeepSpeed is finished, tell the background simulator to shut down
        kill \"\$SIM_PID\" || true
        wait \"\$SIM_PID\" || true
        echo 'Simulation complete.'
    "

echo "Job finished."
