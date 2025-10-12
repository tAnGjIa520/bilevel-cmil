"""
Configuration and argument parsing utilities.

This module provides utilities for loading YAML configuration files
and parsing command-line arguments with support for various data types.
"""

import argparse
import ast
import os
import yaml
import wandb


def parse_list_arg(value_str):
    """
    Parse list arguments from command line.

    Args:
        value_str (str): String representation of a list, e.g., '[1e-4,1e-4,1e-4]' or '[1, 2, 3]'

    Returns:
        Parsed value or original string if parsing fails
    """
    try:
        # Try to safely parse the string using ast.literal_eval
        parsed_value = ast.literal_eval(value_str)
        return parsed_value
    except (ValueError, SyntaxError):
        # If parsing fails, return the original string
        return value_str


def add_argument(parser, name, value):
    """
    Helper function to add an argument to the parser if it doesn't already exist.

    Args:
        parser (argparse.ArgumentParser): Argument parser
        name (str): Argument name
        value: Default value (type will be inferred)
    """
    if not any(arg.dest == name for arg in parser._actions):
        if isinstance(value, bool):
            parser.add_argument(f'--{name}', action='store_false' if value else 'store_true', default=value)
        elif isinstance(value, list):
            # For list types, use custom parsing function
            parser.add_argument(f'--{name}', type=parse_list_arg, default=value)
        else:
            parser.add_argument(f'--{name}', type=type(value), default=value)


def load_config_from_yaml(file_path):
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): Path to YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file) or {}


def init_args():
    """
    Initialize and parse command-line arguments with YAML config support.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='MIL-CL')
    parser.add_argument('--preset', type=str, default=None, help='preset config')

    args, remaining_argv = parser.parse_known_args()

    # Load configuration from YAML if preset is provided
    config = load_config_from_yaml(args.preset) if args.preset else {}

    # Update parser with options from YAML configuration
    for k, v in config.items():
        add_argument(parser, k, v)

    args = parser.parse_args()

    # Update args for debug modes
    if args.debug == 'False':
        args.debug = False
    elif args.debug == 'True':
        args.debug = 'lite'
    if args.debug == 'lite':
        args.n_batches = 10
        args.epochs = 2
        args.n_folds = 1
        args.wandb_mode = 'disabled'
    elif args.debug == 'full':
        args.n_batches = 20
        args.epochs = 3
        args.wandb_mode = 'disabled'

    # Set wandb mode
    os.environ['WANDB_MODE'] = args.wandb_mode
    wandb.require("core")

    # Set fold range defaults
    if args.folds_start == -1:
        args.folds_start = 0
    if args.folds_end == -1:
        if hasattr(args, 'n_folds'):
            args.folds_end = args.n_folds
        else:
            args.folds_end = 1

    print(args)
    return args
