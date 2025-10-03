# decode_hex_to_text.py

def hexfile_to_text(input_file, output_file):
    # Read the file and strip whitespace
    with open(input_file, "r") as f:
        data = f.read()

    # Split into tokens like ["0x7b", "0x22", "0x76", ...]
    hex_values = data.replace("\n", " ").split(",")

    # Convert each hex string to a character
    bytes_list = []
    for h in hex_values:
        h = h.strip()
        if h:  # skip empty entries
            # interpret as hex (remove "0x")
            bytes_list.append(int(h, 16))

    # Build the byte array
    raw_bytes = bytearray(bytes_list)

    # Decode as UTF-8 text
    text = raw_bytes.decode("utf-8")

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Decoded text written to {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python decode_hex_to_text.py input.txt output.json")
    else:
        hexfile_to_text(sys.argv[1], sys.argv[2])
