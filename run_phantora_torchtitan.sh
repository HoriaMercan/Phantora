#!/bin/bash
#SBATCH --job-name=phantora_torchtitan
#SBATCH --output=logs/torchtitan_%j.out
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
EVAL_NNODES="${EVAL_NNODES:-1}"
EVAL_NGPU="${EVAL_NGPU:-2}"
EVAL_VRAM_MIB="${EVAL_VRAM_MIB:-81920}"

# Simulation parameters - set via environment variables for flexibility
# e.g., SIM_NODES=8 SIM_GPUS_PER_NODE=8 ./run_phantora_torchtitan.sh
SIM_NODES="${SIM_NODES:-8}"
SIM_GPUS_PER_NODE="${SIM_GPUS_PER_NODE:-8}"

echo "----------------------------------------------------"
echo "Running Phantora TorchTitan Test Job"
echo "Image: $SIF_IMAGE"
echo "Workspace: $WORKSPACE_DIR"
echo "----------------------------------------------------"

# Make sure the logs directory exists
mkdir -p logs

# Please ensure the tokenizer is downloaded. Note that Llama 3 is gated so you need a Hugging Face token:
# wget --header="Authorization: Bearer YOUR_HF_TOKEN" https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/original/tokenizer.model -O Phantora/tests/assets/tokenizer.model

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
                if [[ -z "\$REAL_CUDART" ]]; then
                        REAL_CUDART=\$(ldconfig -p 2>/dev/null | awk '/libcudart.so.11/ {print \$NF; exit}')
                fi
                if [[ -z "\$REAL_CUDART" ]]; then
                        REAL_CUDART=\$(ldconfig -p 2>/dev/null | awk '/libcudart.so$/ {print \$NF; exit}')
                fi
                if [[ -z "\$REAL_CUDART" ]]; then
                        for candidate in \
                                /.singularity.d/libs/libcudart.so.12 \
                                /.singularity.d/libs/libcudart.so.11 \
                                /.singularity.d/libs/libcudart.so \
                                /usr/lib64/libcudart.so.12 \
                                /usr/lib64/libcudart.so.11 \
                                /usr/lib64/libcudart.so \
                                /usr/lib/x86_64-linux-gnu/libcudart.so.12 \
                                /usr/lib/x86_64-linux-gnu/libcudart.so.11 \
                                /usr/lib/x86_64-linux-gnu/libcudart.so \
                                \$CUDA_HOME/lib64/libcudart.so.12 \
                                \$CUDA_HOME/lib64/libcudart.so.11 \
                                \$CUDA_HOME/lib64/libcudart.so; do
                                if [[ -e "\$candidate" ]]; then
                                        REAL_CUDART="\$candidate"
                                        break
                                fi
                        done
                fi

                if [[ -z "\$REAL_CUDART" ]]; then
                        echo 'ERROR: libcudart not found in container runtime.'
                        exit 1
                fi

                REAL_CUDART_DIR=\$(dirname \"\$REAL_CUDART\")
                BASE_LD_LIBRARY_PATH=\$REAL_CUDART_DIR:\$TORCH_LIB:\$PYTHON_LIB:\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
                export LD_LIBRARY_PATH=\$BASE_LD_LIBRARY_PATH
                export LIBRARY_PATH=\$LD_LIBRARY_PATH

        # export PHANTORA=1
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

                # 5. Generate netconfig from torchtitan template files
                cd /mnt/Phantora
                echo 'Generating netconfig.toml with config_gen.py...'
                python3 tests/docker/torchtitan/config_gen.py \
                        --nhost $SIM_NODES \
                        --ngpu $SIM_GPUS_PER_NODE

                NETCONFIG_FILE=/mnt/Phantora/tests/docker/torchtitan/netconfig.toml

                # For single-node SLURM/apptainer runs, map all simulated hosts to runtime hostname
                if [[ \"\$EVAL_NNODES\" == \"1\" ]]; then
                        THIS_HOST=\$(hostname)
                        # Replace all host-N entries with the real hostname
                        sed -i \"s/host-[0-9]\\+/\$THIS_HOST/g\" \"\$NETCONFIG_FILE\"
                fi
        
                # 6. Start Simulator
                cd /mnt/Phantora/phantora
        echo 'Building computational graph...'
        python3 build_graph.py

        echo 'Starting Phantora Simulator server...'
                PHANTORA_SOCKET_PREFIX=\$PHANTORA_SOCKET_PREFIX \
                LD_PRELOAD= \
                LD_LIBRARY_PATH=\$BASE_LD_LIBRARY_PATH \
                RUST_BACKTRACE=full ./target/release/simulator \
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
        
                # 7. Run TorchTitan
        echo 'Starting TorchTitan training simulation...'
        cd /mnt/Phantora

        # FORCE Python to use the library that WE KNOW has the symbol
        # We point LD_PRELOAD to the one inside the container (/phantora/...)
        PHANTORA_SOCKET_PREFIX=\$PHANTORA_SOCKET_PREFIX \
                PHANTORA_VRAM_MIB=$EVAL_VRAM_MIB \
                PHANTORA_NGPU=$EVAL_NGPU \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                LD_LIBRARY_PATH=/phantora/dist:\$BASE_LD_LIBRARY_PATH \
        LD_PRELOAD=/phantora/dist/libcuda.so.1 /phantora/dist/phantora_run torchrun \\
            --nproc_per_node \$EVAL_NGPU \\
            --nnodes \$EVAL_NNODES \\
            tests/test_torchtitan.py \\
            --job.config_file=tests/test_torchtitan_llama3_8b.toml

        # TorchTitan is finished, tell the background simulator to shut down
        kill "\$SIM_PID" || true
        wait "\$SIM_PID" || true
        echo 'Simulation complete.'
    "

echo "Job finished."
