#!/bin/bash

# We start by downloading the images
# Check if uhd_images_downloader is available
if ! command -v uhd_images_downloader >/dev/null 2>&1; then
    echo "Error: could not find 'uhd_images_downloader'."
    echo "Is UHD installed?"
    exit 1
fi

# If found, print the path for confirmation
echo "Running image downloader for B200-series..."

# Run the tool and capture output
output=$(uhd_images_downloader --types b2.* 2>&1)

# Check exit code
if [ $? -ne 0 ]; then
    echo "Error: uhd_images_downloader failed!"
    echo "$output"
    exit 2
fi

# Extract the images path from the output
# Example line: [INFO] Images destination: /usr/share/uhd/4.8.0/images
images_dir=$(echo "$output" | grep -oP '(?<=\[INFO\] Images destination: ).*')

if [ -z "$images_dir" ]; then
    echo "Warning: Could not find images directory in output."
else
    echo "Images will be installed at: $images_dir"
fi
# Now we can use $images_dir later in the script
# Example:
# export UHD_IMAGES_DIR="$images_dir"


# Next we check if the USB udev rules are installed.
# If not, we install them.

RULE_FILE="/etc/udev/rules.d/99-usrp.rules"

# Check if the file already exists
if [ -f "$RULE_FILE" ]; then
    echo "$RULE_FILE already exists. Skipping creation."
else
    echo "Creating $RULE_FILE with USRP B200/B210 rules..."
    
    sudo tee "$RULE_FILE" > /dev/null << 'EOF'
# Ettus Research USRP B200/B210
SUBSYSTEM=="usb", ATTR{idVendor}=="2500", ATTR{idProduct}=="0020", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="2500", ATTR{idProduct}=="0021", MODE="0666", GROUP="plugdev"
EOF

    # Reload udev rules
    echo "Reloading udev rules..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger

    echo "Done. Please unplug and replug your USRP device."
fi

# Add the necessary lines to .bashrc
ROOT_BASHRC="/root/.bashrc"
PI_BASHRC="/home/pi/.bashrc"

# Lines to ensure exist, using the variable
LINE1="export UHD_IMAGES_DIR=\"$images_dir\""
LINE2='export PYTHONPATH="/usr/local/lib/python3.11/site-packages:$PYTHONPATH"'

# Function: append line if no line starts with the same variable
append_if_missing_prefix() {
    local bashrc="$1"      # first argument = bashrc file
    local line="$2"        # second argument = line to append
    local var_name=$(echo "$line" | cut -d '=' -f 1)

    if ! grep -q "^$var_name" "$bashrc"; then
        echo "$line" >> "$bashrc"
        echo "Added line to $bashrc: $line"
    else
        echo "Variable '$var_name' already defined in $bashrc"
    fi
}

# Check each line
append_if_missing_prefix "$ROOT_BASHRC" "$LINE1"
append_if_missing_prefix "$PI_BASHRC" "$LINE1"
#append_if_missing_prefix "$LINE2"

python3 - <<'EOF'
import os, sys
try:
    import uhd
    print("✅ UHD module loaded from:", uhd.__file__)
    usrp = uhd.usrp.MultiUSRP()
    print("✅ MultiUSRP() created successfully!")
except Exception as e:
    print("❌ First attempt failed:", e)
    print("🔁 Retrying with custom PYTHONPATH...")
    os.environ["PYTHONPATH"] = "/usr/local/lib/python3.11/site-packages:" + os.environ.get("PYTHONPATH", "")
    try:
        import importlib; importlib.invalidate_caches()
        import uhd
        print("✅ UHD module reloaded from:", uhd.__file__)
        usrp = uhd.usrp.MultiUSRP()
        print("✅ MultiUSRP() created successfully after setting PYTHONPATH!")
    except Exception as e2:
        print("❌ Still failed:", e2)
        sys.exit(1)
EOF

echo "Done checking .bashrc."



echo "OK"   # output "OK" is captured by ansible
exit 0