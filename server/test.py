from utils.ansible_utils import get_target_hosts
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
inventory=os.path.join(project_dir, 'ansible/inventory/hosts.yaml')

hosts = get_target_hosts(inventory, limit="A05 A06")

print("Hosts:", [h.name for h in hosts])
print("Count:", len(hosts))
