import os
from ansible.parsing.dataloader import DataLoader
from ansible.inventory.manager import InventoryManager
from ansible.vars.manager import VariableManager


# Disable noisy Ansible warnings (safe subset)
def disable_ansible_warnings():
    os.environ["ANSIBLE_INVENTORY_UNPARSED_WARNING"] = "False"
    os.environ["ANSIBLE_INVENTORY_ANY_UNPARSED_IS_FAILED"] = "False"
    os.environ["ANSIBLE_DEPRECATION_WARNINGS"] = "False"
    os.environ["ANSIBLE_INVALID_TASK_ATTRIBUTE_FAILED"] = "False"
    os.environ["ANSIBLE_HOSTS_WARNING"] = "False"


def get_target_hosts(inventory_path, limit=None, suppress_warnings=True):
    """
    Returns a list of ansible.inventory.host.Host objects targeted
    by an inventory (optionally filtered by limit/group/hostname).

    Args:
        inventory_path (str): path to hosts.yaml
        limit (str): host/group pattern ("A05", "edge-nodes", etc.)
        suppress_warnings (bool): disable warnings automatically

    Returns:
        list[Host]: result of InventoryManager.get_hosts()
    """

    if suppress_warnings:
        disable_ansible_warnings()

    loader = DataLoader()
    inventory = InventoryManager(loader=loader, sources=[inventory_path])
    variable_manager = VariableManager(loader=loader, inventory=inventory)

    if limit:
        inventory.subset(limit)

    return inventory.get_hosts()
