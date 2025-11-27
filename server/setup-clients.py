from utils.ansible_utils import get_target_hosts
import ansible_runner
import shutil
import os

# Some basic config
tiles = "A05 A06 A07 A08 A09 A10" # hosts (clients) to set up
#tiles = "ceiling" # hosts (clients) to set up
repository_name = 'geometry-based-wireless-power-transfer'  # Name of the git repository
organisation_name = 'techtile-by-dramco'

# Rest of the configuration we obtain automatically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
inventory_path = os.path.join(project_dir, 'ansible/inventory/hosts.yaml')
host_list = get_target_hosts(inventory_path, limit=tiles, suppress_warnings=True)

print("Project directory: ", project_dir)
print("Targeting", len(host_list), "hosts:", [h.name for h in host_list])

venv_path = os.path.join(project_dir, "server")
venv_bin_path = os.path.join(venv_path, "bin")

env = os.environ.copy()
env["PATH"] = f"{venv_bin_path}:{env['PATH']}"  # ensures runner uses venv ansible-playbook
env["VIRTUAL_ENV"] = venv_path

try:
    # Pull the repo on the client
    r = ansible_runner.run(
        private_data_dir=project_dir,
        playbook=os.path.join(project_dir, 'ansible/server/pull-repo.yaml'), # paths relative to private_data_dir, didn't seem to work
        inventory=inventory_path,
        extravars={
            'org_name': organisation_name,
            'repo_name': repository_name,
        },
        limit=tiles,
        envvars=env
    )

    # Print the status and return code
    print("Status:", r.status)   # e.g. 'successful' or 'failed'
    print("Return code:", r.rc)

    # Optionally, print stdout of all tasks
    for event in r.events:
        if 'stdout' in event:
            print(event['stdout'])
    
    if r.rc != 0:
        raise RuntimeError(f"Ansible playbook failed with return code {r.rc}")
    
    # Now that we have the repo on the client, we can run scripts
    # First thing we do is check if UHD is up-and-running
    r = ansible_runner.run(
        private_data_dir=project_dir,
        playbook=os.path.join(project_dir, 'ansible/server/run-script.yaml'),
        inventory=inventory_path,
        extravars={
            'script_path': os.path.join('/home/pi/', repository_name, 'ansible/tiles/check-uhd.sh'),
            'sudo': 'yes',
            'sudo_flags': '-E'
        },
        limit=tiles,
        envvars=env
    )

    # Print the status and return code
    print("Status:", r.status)   # e.g. 'successful' or 'failed'
    print("Return code:", r.rc)

    # Optionally, print stdout of all tasks
    for event in r.events:
        if 'stdout' in event:
            print(event['stdout'])

    if r.rc != 0:
        raise RuntimeError(f"Ansible playbook failed with return code {r.rc}")

    # Todo (optionally): run script (on the client) that sets up venv and downloads necessary packages through pip

except FileNotFoundError as e:
    print(f"File not found: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    # cleanup directories created by ansible-runner
    shutil.rmtree(os.path.join(project_dir, "artifacts"), ignore_errors=True)
    shutil.rmtree(os.path.join(project_dir, "env"), ignore_errors=True)