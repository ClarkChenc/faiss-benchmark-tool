import argparse
import os
import numpy as np
from faiss_benchmark.utils import fvecs_read, ivecs_read, fvecs_write, ivecs_write

def split_file(input_path, output_path, count):
    """Splits a .fvecs or .ivecs file to a specified number of entries."""
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Reading from {input_path}...")
    if input_path.endswith('.fvecs'):
        data = fvecs_read(input_path)
    elif input_path.endswith('.ivecs'):
        data = ivecs_read(input_path)
    else:
        print(f"Error: Unsupported file format for {input_path}. Only .fvecs and .ivecs are supported.")
        return

    if count > len(data):
        print(f"Warning: Requested count ({count}) is larger than the number of entries in the file ({len(data)}). Using all entries.")
        count = len(data)

    print(f"Splitting to {count} entries...")
    split_data = data[:count]

    print(f"Writing to {output_path}...")
    if output_path.endswith('.fvecs'):
        fvecs_write(output_path, split_data)
    elif output_path.endswith('.ivecs'):
        ivecs_write(output_path, split_data)
    else:
        print(f"Error: Unsupported file format for {output_path}. Only .fvecs and .ivecs are supported.")
        return

    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Split .fvecs and .ivecs files.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input file (.fvecs or .ivecs)")
    parser.add_argument("-o", "--output", required=True, help="Path to the output file (.fvecs or .ivecs)")
    parser.add_argument("-c", "--count", required=True, type=int, help="Number of entries to keep")
    args = parser.parse_args()

    split_file(args.input, args.output, args.count)

if __name__ == "__main__":
    main()