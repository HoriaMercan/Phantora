#!/usr/bin/env python3

SIMULATOR_TEMPLATE = r"""
  simulator:
    image: "phantora:latest"
    volumes:
      - /run/phantora:/run/phantora
      - ./netconfig.toml:/netconfig.toml:ro
    pid: host
    ipc: host
    environment:
      - PHANTORA_LOG=${{PHANTORA_LOG:-info}}
      - PHANTORA_SOCKET_PREFIX=/run/phantora/phantora
    command: /phantora/dist/phantora_server --netconfig /netconfig.toml
    cpuset: '{cpuset}'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
"""

HOST_TEMPLATE = r"""
  host-{host_id}:
    image: "phantora:latest"
    volumes:
      - /run/phantora:/run/phantora
      - ../..:/phantora/tests:ro
    pid: host
    ipc: host
    environment:
      - NGPU={ngpu}
      - PHANTORA_NGPU={ngpu}
      - PHANTORA_VRAM_MIB={vram_mib}
      - PHANTORA_SOCKET_PREFIX=/run/phantora/phantora
    hostname: host-{host_id}
    command: sleep infinity
    cpuset: '{cpuset}'
    depends_on:
      - simulator
"""

NETCONFIG_TEMPLATE = r"""
host_mapping = {host_list}

[simulator]
loopback_speed = 2880
fairness = "PerFlowMaxMin"
{custom_model_line}
[topology]
type = "TwoLayerMultiPath"

[topology.args]
nspines = {nspines}
nracks = {nracks}
rack_size = {rack_size}
host_bw = 400
rack_uplink_port_bw = 400
load_balancer_type = "EcmpEverything"
"""

if __name__ == '__main__':
    import argparse
    from os.path import dirname, realpath, join
    from multiprocessing import cpu_count
    script_dir = dirname(realpath(__file__))

    nproc = cpu_count()
    if nproc <= 2:
        default_sim_core = str(nproc - 1)
        default_host_cpuset = str(nproc - 1)
    else:
        default_sim_core = str(nproc // 2)
        default_host_cpuset = f"{nproc // 2 + 1}-{nproc - 1}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--nhost", type=int, default=4)
    parser.add_argument("--ngpu", type=int, default=4)
    parser.add_argument("--vram_mib", type=int, default=143771)
    parser.add_argument("--cpuset_sim", type=str, default=default_sim_core)
    parser.add_argument("--cpuset_host", type=str, default=default_host_cpuset)
    parser.add_argument("--custom_model", type=str, default="")
    args = parser.parse_args()

    nhosts = args.nhost
    ngpu = args.ngpu

    with open(join(script_dir, "compose.yaml"), "w") as f:
      f.write("services:")
      f.write(SIMULATOR_TEMPLATE.format(cpuset=args.cpuset_sim))
      for i in range(1, nhosts + 1):
          f.write(HOST_TEMPLATE.format(
              host_id=i, ngpu=ngpu, vram_mib=args.vram_mib, cpuset=args.cpuset_host
          ))

    with open(join(script_dir, "netconfig.toml"), "w") as f:
        host_list = str([f"host-{i}" for i in range(1, nhosts + 1)])
        custom_model_line = f'custom_model_path = "{args.custom_model}"\n' if args.custom_model else ""
        
        # Dynamic topology calculation for realistic cluster simulation
        # Two strategies based on cluster size:
        # 
        # STRATEGY 1: Local/Lab clusters (nhosts <= 4)
        #   - Pack nodes together: mimics single cabinet deployments
        #   - rack_size = 8 (standard cabinet)
        #   - nracks = 1 (everything in one rack)
        #   - Good for: testing on limited hardware
        #
        # STRATEGY 2: Large/Distributed clusters (nhosts > 4)
        #   - Spread nodes across racks: models real data center distribution
        #   - rack_size = 1 (one node per rack, more realistic for scale)
        #   - nracks = nhosts (each node gets its own rack/cabinet)
        #   - Good for: simulating realistic network behavior across sites
        
        if nhosts <= 4:
            # Strategy 1: Colocate in single cabinet
            rack_size = 8
            nracks = 1
        else:
            # Strategy 2: Distribute across racks
            rack_size = 1
            nracks = nhosts
        
        # Spine scaling based on cluster size and topology
        # - Intra-rack: minimal spines needed (within cabinet)
        # - Inter-rack: more spines needed for cross-rack bandwidth
        if nhosts <= 4:
            nspines = 2  # Small cluster, basic redundancy
        elif nhosts <= 16:
            nspines = 4  # Medium cluster, good cross-rack connectivity
        elif nhosts <= 64:
            nspines = 8  # Large cluster, high-capacity core
        else:
            nspines = max(16, nhosts // 4)  # Very large, scales with cluster
        
        f.write(NETCONFIG_TEMPLATE.format(host_list=host_list, nracks=nracks, nspines=nspines, rack_size=rack_size, custom_model_line=custom_model_line))

    with open(join(script_dir, "config.sh"), "w") as f:
        f.write(f"EVAL_NHOST={nhosts}\n")
        f.write(f"EVAL_NGPU={ngpu}\n")
