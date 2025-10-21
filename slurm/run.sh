#!/bin/bash
set +e  # Don't exit on error
torchrun --nproc_per_node=8 your_training_script.py
echo "Exit code: $?"
exec bash  # Drop to interactive shell after completion
