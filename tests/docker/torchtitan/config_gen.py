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
        # For larger clusters, scale topology: more spines for better load balancing
        # nspines = 2 for small clusters (2-4 nodes), 4 for medium (8 nodes), 8+ for larger
        nspines = 2 if nhosts <= 4 else (4 if nhosts <= 16 else 8)
        # rack_size: max 8 per rack (typical A100 cabinet size); keeps locality for allreduce
        rack_size = nhosts if nhosts <= 8 else 8
        # nracks: fit hosts into racks; e.g., 8 hosts in 8-sized rack = 1 rack, 16 hosts = 2 racks
        nracks = (nhosts + rack_size - 1) // rack_size
        f.write(NETCONFIG_TEMPLATE.format(host_list=host_list, nracks=nracks, nspines=nspines, rack_size=rack_size))

    with open(join(script_dir, "config.sh"), "w") as f:
        f.write(f"EVAL_NHOST={nhosts}\n")
        f.write(f"EVAL_NGPU={ngpu}\n")
