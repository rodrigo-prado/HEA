#!/usr/bin/env python3

import argparse


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Conflict Graph Creation.')
    parser.add_argument('input_file',
                        action='store',
                        metavar='INPUT_FILE',
                        type=str,
                        help='Input Workflow DAG file.')
    parser.add_argument('output_file',
                        action='store',
                        metavar='OUTPUT_FILE',
                        type=str,
                        help='Output Conflict Graph File.')
    arg = parser.parse_args()

    # Reading Input File
    with open(arg.input_file, mode='r') as file:
        print(file)


if __name__ == '__main__':
    main()
