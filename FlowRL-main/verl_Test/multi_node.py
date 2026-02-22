import os
import socket
import subprocess
import time

COMMAND = os.environ.get('COMMAND', 'bash')
SCRIPT = os.environ.get('SCRIPT', '/fs-computility/plm/shared/zhuxuekai/reasoning_flow/verl/command/aiyun/baselines/32B/rpp_32b_qwen.sh')

def get_master_address(master_port):
    """
    获取头节点的地址。
    通过 socket 解析主机名并返回 IP 地址。
    """
    ip_address = socket.gethostbyname(os.environ.get("MLP_WORKER_0_HOST"))
    return f"{ip_address}:{master_port}"

def start_ray_head_node():
    """
    在头节点上启动 Ray 集群。
    """
    print("Starting Ray head node...")
    subprocess.run(["ray", "start", "--head"], check=True)

def connect_to_ray_cluster(master_address):
    """
    在工作节点上连接到 Ray 集群。
    """
    print(f"Connecting to Ray cluster at {master_address}...")
    subprocess.run(["ray", "start", "--address", master_address], check=True)

def execute_entry_command():
    """
    执行指定的入口命令。
    """
    print(F"Executing entry command: {COMMAND} {SCRIPT}")
    subprocess.run([COMMAND, SCRIPT], check=True)

def main():
    # 获取环境变量
    rank = int(os.environ.get("MLP_ROLE_INDEX", -1))
    # master_port = int(os.environ.get("MASTER_PORT", 6379))

    if rank == 0:
        # 头节点：启动 Ray 集群并执行入口命令
        start_ray_head_node()
        master_address = get_master_address(6379)
        print(f"Ray head node started. Master address: {master_address}")
        time.sleep(20)
        execute_entry_command()
    else:
        # 工作节点：获取头节点地址并连接到 Ray 集群
        master_address = get_master_address(6379)
        connect_to_ray_cluster(master_address)
        print(f"Worker node connected to Ray cluster at {master_address}")

        # 工作节点进入等待状态，等待任务分配
        while True:
            time.sleep(60)  # 持续运行，避免退出

if __name__ == "__main__":
    main()