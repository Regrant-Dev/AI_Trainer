import json
import os
import sys

# Function to read data from a JSON file
def read_data_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to convert data to JSONL format
def convert_to_jsonl(data):
    jsonl_data = []
    for entry in data:
        instruction = entry['prompt']
        system_message = None
        user_message = None
        assistant_message = None
        for message in entry['conversations']:
            if message['role'] == 'system':
                system_message = message['content']
            elif message['role'] == 'user':
                user_message = message['content']
            elif message['role'] == 'assistant':
                assistant_message = message['content']
        if system_message and user_message and assistant_message:
            jsonl_data.append({
                "prompt": instruction,
                "conversations": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
            })
    return jsonl_data

# Function to save data to a JSONL file
def save_data_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Main function to process input and output file paths
def main(input_filename):
    data = read_data_from_file(input_filename)
    jsonl_data = convert_to_jsonl(data)
    
    input_dir = os.path.dirname(input_filename)
    output_filename = os.path.join(input_dir, 'converted_data.jsonl')
    
    save_data_to_jsonl(jsonl_data, output_filename)
    print(f"Data successfully converted to JSONL format and saved to {output_filename}")

# Ensure script is run with appropriate arguments
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_jsonl.py <input_file>")
    else:
        input_filename = sys.argv[1]
        main(input_filename)
