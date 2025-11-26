import ansible_runner
import shutil
import os

# Some basic config
hosts = 'A05' # hosts (clients) to set up
repository_name = 'geometry-based-wireless-power-transfer'  # Name of the git repository

# Rest of the configuration we obtain automatically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

print("Project dir: ", project_dir)
print("Working on tile(s):", hosts)

try:
    # Pull the repo on the client
    r = ansible_runner.run(
        private_data_dir=project_dir,
        playbook=os.path.join(project_dir, 'ansible/server/clean-home.yaml'), # paths relative to private_data_dir, didn't seem to work
        inventory=os.path.join(project_dir, 'ansible/inventory/hosts.yaml'),
        extravars={
            'repo_name': repository_name,
        },
        limit=hosts
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