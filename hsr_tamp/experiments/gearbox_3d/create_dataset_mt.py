import subprocess

num_count = 10000
for i in range(num_count):
    # Execute create_dataset.py
    result = subprocess.run(['python3', 'create_dataset.py', '--skill'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get results from create_dataset.py
    stdout = result.stdout.decode('utf-8')
    stderr = result.stderr.decode('utf-8')

    # If finish with no error
    if result.returncode == 0:
        print("create_dataset.py completed successfully.")
    else:
        # Prin error message
        print("create_dataset.py encountered an error:")
        print(stderr)

    #  Rerun create_dataset.py
    print("Restarting create_dataset.py...")