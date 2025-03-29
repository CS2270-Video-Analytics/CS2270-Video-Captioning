#!/bin/bash

# Correctly setting the SINGULARITY_CACHEDIR environment variable
export SINGULARITY_CACHEDIR="/users/$USER/scratch"

# Sourcing .bashrc to apply any other changes if necessary
source ~/.bashrc

# Running the Singularity build command
singularity build cs2270.simg --docker-image shreyasraman/cs2270:courses