#!/usr/bin/env python3
"""
Ray Cluster Setup Script (Docker/default-env)

- No conda activation; assumes 'ray' and 'python' are in PATH on all nodes
- Resolves head IP via SSH so local DNS is not required (or use HEAD_IP env)
- Starts head locally if on head, otherwise starts it remotely
- Optionally aligns Ray versions across nodes
- Lets you pick nodes via --nodes "host0,host1" or --num-nodes N
"""

import subprocess
import time
import sys
import os
import socket
from typing import List, Dict, Tuple

# ===================== Defaults (override with --nodes/--num-nodes) =====================
DEFAULT_NODES = [
    "ttt-mpiworker-0",  # Head
    "ttt-mpiworker-1",  # First worker
    "ttt-mpiworker-2",
    "ttt-mpiworker-3",
]

# Set to None to skip enforcing a specific Ray version
RAY_VERSION = None  # e.g., "2.47.1" or "2.48.0"

# Ray ports
RAY_PORT = 6379
DASHBOARD_PORT = 8265
OBJECT_MANAGER_PORT = 8076
NODE_MANAGER_PORT = 8077
CLIENT_SERVER_PORT = 10001  # ray client port

# SSH options
SSH_COMMON = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
]

# ---------- small helpers ----------

def run_local(cmd: List[str] | str, timeout: int = 60, shell: bool = False):
    return subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout)

def run_remote(host: str, raw_cmd: str, timeout: int = 60):
    """Run a command on a remote host. No conda activation; default env is assumed."""
    # Use bash -lc so login PATH/rc are applied if needed; no conda!
    wrapped = f'bash -lc {quote_sh(raw_cmd)}'
    cmd = ["ssh", *SSH_COMMON, host, wrapped]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def quote_sh(s: str) -> str:
    # minimal safe quoting for bash -lc
    return "'" + s.replace("'", "'\"'\"'") + "'"

def ok(msg): print(f"  ✓ {msg}")
def warn(msg): print(f"  ⚠ {msg}")
def err(msg): print(f"  ✗ {msg}")

def resolve_head_ip_via_ssh(head_alias: str) -> str:
    # 0) env override
    env_ip = os.environ.get("HEAD_IP", "").strip()
    if env_ip:
        return env_ip
    # 1) try DNS
    try:
        return socket.gethostbyname(head_alias)
    except Exception:
        pass
    # 2) ask head for src IP to 8.8.8.8
    try:
        r = run_remote(head_alias, "ip route get 8.8.8.8 | awk '{print $7; exit}'")
        ip = (r.stdout or "").strip()
        if r.returncode == 0 and ip:
            return ip
    except Exception:
        pass
    # 3) fallback hostname -I
    try:
        r = run_remote(head_alias, "hostname -I | awk '{print $1}'")
        ip = (r.stdout or "").strip()
        if r.returncode == 0 and ip:
            return ip
    except Exception:
        pass
    raise RuntimeError(f"Cannot resolve IP for head '{head_alias}'. Set HEAD_IP=<ip> and retry.")

def get_ray_version_local() -> Tuple[bool, str]:
    r = run_local(["python", "-c", "import ray; print(ray.__version__)"])
    return (r.returncode == 0, (r.stdout or r.stderr).strip())

def get_ray_version_remote(host: str) -> Tuple[bool, str]:
    r = run_remote(host, "python -c 'import ray; print(ray.__version__)'")
    return (r.returncode == 0, (r.stdout or r.stderr).strip())

def ensure_ray_version_remote(host: str, version: str) -> bool:
    ok_now, ver = get_ray_version_remote(host)
    if ok_now and ver.strip() == version:
        ok(f"Ray {version} already on {host}")
        return True
    warn(f"Installing Ray {version} on {host} (was: {ver if ok_now else 'missing'})")
    r = run_remote(host, f'pip install -U "ray[default]=={version}"', timeout=600)
    if r.returncode != 0:
        err(f"pip install ray on {host} failed:\n{(r.stderr or r.stdout).strip()[:300]}")
        return False
    ok2, ver2 = get_ray_version_remote(host)
    if ok2 and ver2.strip() == version:
        ok(f"Ray {version} present on {host}")
        return True
    err(f"Ray version check failed on {host}: {ver2}")
    return False

class RayClusterManager:
    def __init__(self, nodes: List[str]):
        assert len(nodes) >= 1, "Need at least head node"
        self.nodes = nodes
        self.head_node = nodes[0]
        self.worker_nodes = nodes[1:]
        self.current_hostname = socket.gethostname()
        self.head_ip = resolve_head_ip_via_ssh(self.head_node)

        if self.head_node not in self.current_hostname:
            warn(f"Warning: this script is ideally run on {self.head_node}")
            print(f"  Current hostname: {self.current_hostname}")
            if input("Continue anyway? (y/n): ").lower() != "y":
                sys.exit(1)

    def test_ssh_connectivity(self) -> Dict[str, bool]:
        print("\n📡 Testing SSH connectivity to worker nodes...")
        connectivity: Dict[str, bool] = {}
        for node in self.worker_nodes:
            try:
                r = run_remote(node, "echo connected", timeout=15)
                if r.returncode == 0:
                    ok(f"SSH to {node} successful")
                    connectivity[node] = True
                else:
                    err(f"SSH to {node} failed: {(r.stderr or r.stdout).strip()[:160]}")
                    connectivity[node] = False
            except Exception as e:
                err(f"SSH to {node} failed: {e}")
                connectivity[node] = False
        return connectivity

    def check_ray_installed(self) -> Dict[str, bool]:
        print("\n🔍 Checking Ray installation on all nodes...")
        status: Dict[str, bool] = {}

        ok_local, ver_local = get_ray_version_remote(self.head_node) \
                              if self.head_node not in self.current_hostname else get_ray_version_local()
        if ok_local:
            ok(f"Ray installed on {self.head_node} (version: {ver_local.strip()})")
        else:
            err(f"Ray not installed on {self.head_node}: {ver_local}")
        status[self.head_node] = ok_local

        for node in self.worker_nodes:
            ok_r, ver_r = get_ray_version_remote(node)
            if ok_r:
                ok(f"Ray installed on {node} (version: {ver_r.strip()})")
            else:
                err(f"Ray not installed on {node}: {ver_r.strip()}")
            status[node] = ok_r

        return status

    def align_ray_versions(self) -> bool:
        if not RAY_VERSION:
            return True
        print(f"\n🧩 Ensuring Ray=={RAY_VERSION} on all nodes...")
        all_ok = True

        # Head
        if self.head_node in self.current_hostname:
            ok_local, ver_local = get_ray_version_local()
            if not ok_local or ver_local.strip() != RAY_VERSION:
                warn(f"Installing Ray {RAY_VERSION} on {self.head_node} (was: {ver_local if ok_local else 'missing'})")
                r = run_local(['pip', 'install', '-U', f'ray[default]=={RAY_VERSION}'], timeout=600)
                if r.returncode != 0:
                    err(f"pip install on head failed: {r.stderr.strip()[:300]}")
                    all_ok = False
        else:
            if not ensure_ray_version_remote(self.head_node, RAY_VERSION):
                all_ok = False

        # Workers
        for node in self.worker_nodes:
            if not ensure_ray_version_remote(node, RAY_VERSION):
                all_ok = False

        return all_ok

    def start_head_node(self) -> bool:
        print(f"\n🚀 Starting Ray head node on {self.head_node}...")
        print(f"  Head node IP: {self.head_ip}")

        cmd_stop = "ray stop --force || true"
        cmd_start = (
            "ray start --head "
            f"--port={RAY_PORT} "
            f"--dashboard-host=0.0.0.0 --dashboard-port={DASHBOARD_PORT} "
            f"--object-manager-port={OBJECT_MANAGER_PORT} "
            f"--node-manager-port={NODE_MANAGER_PORT} "
            f"--ray-client-server-port={CLIENT_SERVER_PORT} "
            "--include-dashboard=true"
        )

        if self.head_node in self.current_hostname:
            run_local(cmd_stop, shell=True)
            time.sleep(1)
            r = run_local(cmd_start, shell=True, timeout=60)
        else:
            run_remote(self.head_node, cmd_stop, timeout=60)
            time.sleep(1)
            r = run_remote(self.head_node, cmd_start, timeout=60)

        if r.returncode == 0:
            ok("Ray head node started successfully")
            print(f"  📊 Dashboard: http://{self.head_ip}:{DASHBOARD_PORT}")
            return True
        err("Failed to start Ray head node")
        sys.stderr.write((r.stderr or r.stdout)[-500:] + "\n")
        return False

    def start_worker_node(self, worker: str) -> bool:
        print(f"  Starting Ray worker on {worker}...")
        run_remote(worker, "ray stop --force 2>/dev/null || true", timeout=60)
        time.sleep(0.5)
        r = run_remote(worker, f"ray start --address='{self.head_ip}:{RAY_PORT}'", timeout=120)
        if r.returncode == 0:
            ok(f"Ray worker started on {worker}")
            return True
        err(f"Failed to start Ray worker on {worker}")
        if r.stdout: print("      STDOUT:", r.stdout.strip().splitlines()[-1][:200])
        if r.stderr: print("      STDERR:", r.stderr.strip().splitlines()[-1][:200])
        return False

    def start_cluster(self) -> bool:
        print("=" * 70)
        print("🚀 RAY CLUSTER INITIALIZATION")
        print("=" * 70)

        connectivity = self.test_ssh_connectivity()
        installed = self.check_ray_installed()

        if self.worker_nodes and not all(connectivity.values()):
            warn("Some worker nodes are not accessible via SSH")
            bad = [n for n, okk in connectivity.items() if not okk]
            print(f"  Inaccessible: {', '.join(bad)}")
            if input("Continue with accessible nodes only? (y/n): ").lower() != "y":
                return False

        if not self.align_ray_versions():
            warn("Version alignment encountered issues; proceeding anyway")

        if not self.start_head_node():
            print("❌ Failed to start head node. Aborting.")
            return False

        if self.worker_nodes:
            print(f"\n🔧 Starting {len(self.worker_nodes)} worker nodes...")
            good = 0
            bad: List[str] = []
            for w in self.worker_nodes:
                if connectivity.get(w, False) and installed.get(w, True):
                    if self.start_worker_node(w):
                        good += 1
                    else:
                        bad.append(w)
                else:
                    warn(f"Skipping {w} (not accessible or Ray not present)")
                time.sleep(0.2)

        print("\n" + "=" * 70)
        print("📊 CLUSTER STATUS SUMMARY")
        print("=" * 70)
        print(f"  ✓ Head Node: {self.head_node} ({self.head_ip})")
        if self.worker_nodes:
            print(f"  ✓ Workers Attempted: {len(self.worker_nodes)}")
        print(f"\n  🌐 Ray Address: ray://{self.head_ip}:{CLIENT_SERVER_PORT}")
        print(f"  📊 Dashboard: http://{self.head_ip}:{DASHBOARD_PORT}")
        print("=" * 70)

        return True

    def stop_cluster(self):
        print("\n🛑 Stopping Ray cluster...")
        for w in self.worker_nodes:
            print(f"  Stopping Ray on {w}...")
            run_remote(w, "ray stop --force 2>/dev/null || true", timeout=60)
        print(f"  Stopping Ray on {self.head_node}...")
        if self.head_node in self.current_hostname:
            run_local("ray stop --force 2>/dev/null || true", shell=True, timeout=60)
        else:
            run_remote(self.head_node, "ray stop --force 2>/dev/null || true", timeout=60)
        print("✓ Ray cluster stopped.")

    def get_cluster_status(self):
        print("\n" + "=" * 70)
        print("📊 RAY CLUSTER STATUS")
        print("=" * 70)
        r = run_remote(self.head_node, "ray status")
        if r.returncode == 0:
            print("\n📍 Cluster Overview:")
            print(r.stdout)
        else:
            err("Ray is not running on the head node")
            print("  Try: ssh", self.head_node, "'ray status'")
            return

    def run_test_job(self):
        print("\n🧪 Running test job on Ray cluster...")
        print("-" * 70)
        try:
            import ray
            ray.init(address=f"ray://{self.head_ip}:{CLIENT_SERVER_PORT}", namespace="verl")
            print("📊 Cluster Resources:", ray.cluster_resources())
            @ray.remote
            def f(i):
                import socket, time
                time.sleep(0.1)
                return f"hi {i} from {socket.gethostname()}"
            rs = ray.get([f.remote(i) for i in range(8)])
            for line in rs:
                print("  " + line)
            ray.shutdown()
            print("\n✓ Test job completed successfully!")
        except Exception as e:
            err(f"Test job failed: {e}")
            print("  Make sure the cluster is running: python ray_cluster.py start")

# ---------- CLI ----------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Ray Cluster Manager (Docker/default env)")
    p.add_argument("action", choices=["start", "stop", "status", "test"])
    p.add_argument("--nodes", type=str, default=None,
                   help="Comma-separated hostnames (first is head). Overrides --num-nodes.")
    p.add_argument("--num-nodes", type=int, default=2,
                   help="Use the first N hosts from DEFAULT_NODES (default: 2 => head+first worker)")
    args = p.parse_args()

    if args.nodes:
        nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    else:
        n = max(1, min(args.num_nodes, len(DEFAULT_NODES)))
        nodes = DEFAULT_NODES[:n]

    mgr = RayClusterManager(nodes)

    if args.action == "start":
        if mgr.start_cluster():
            print("\n✅ Ray cluster started.")
            print(f"Nodes: {', '.join(nodes)}")
            print(f"Connect with: ray.init(address='ray://{mgr.head_ip}:{CLIENT_SERVER_PORT}', namespace='verl')")
        else:
            sys.exit(1)
    elif args.action == "stop":
        mgr.stop_cluster()
    elif args.action == "status":
        mgr.get_cluster_status()
    elif args.action == "test":
        mgr.run_test_job()

if __name__ == "__main__":
    main()

