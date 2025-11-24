# A "new" experiment control system
In the new system, ansible is used to manage the fleet of raspberry pi's that are part of the experiment setup.

A typical experiment can be divided into 3 phases:
1. A general system setup.
    a. Make sure the rpi OS and software is up-to-date.
    b. Pull the experiment repo and check that all experiment-specific requirements are met.

2. Run the experiment (can be an iterative process)
    a. Pull the repo (if changes to certain run settings have been made).
    b. Run the experiment scripts on the rpi's (output can be registered asynchronously)
    c. Don't forget to collect your results

3. Clean-up the pi (optional but highly encouraged)

TODO: Detailed description