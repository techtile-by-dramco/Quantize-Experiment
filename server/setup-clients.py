import ansible_runner

# Path to the directory containing your playbook and inventory
base_dir_name = 'geometry-based-wireless-power-transfer'
private_data_dir = '~/' + base_dir_name
hosts = 'A05'

try:
    # Pull the repo
    r = ansible_runner.run(
        private_data_dir=private_data_dir,
        playbook='ansible/server/pull-repo.yaml',
        inventory='ansible/inventory/hosts.yaml',
        extravars={
            'repo_name': 'https://github.com/techtile-by-dramco/' + base_dir_name,
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
            
    # Check if UHD is up-and-running
    r = ansible_runner.run(
        private_data_dir=private_data_dir,
        playbook='ansible/server/run-script.yaml',
        inventory='ansible/inventory/hosts.yaml',
        extravars={
            'script_path': '/home/pi/' + base_dir_name + '/ansible/tiles/check-uhd.sh',
            'sudo': 'yes',
            'sudo_flags': '-E'
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

except FileNotFoundError as e:
    print(f"File not found: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")