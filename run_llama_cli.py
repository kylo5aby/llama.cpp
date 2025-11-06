#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse

model_path = "~/zhenwei/data/models/qwen2-7b-instruct-q4_0.gguf"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run llama-cli with configurable parameters')
    
    parser.add_argument('-length', '--length', 
                       choices=['1024', '4096', '8192'],
                       required=True,
                       help='Prompt length: 1024, 4096, or 8192')
    
    parser.add_argument('-type', '--type',
                       choices=['prefill', 'buffer', 'GDS'],
                       required=True,
                       help='Type: prefill, buffer, or GDS')
    
    return parser.parse_args()

def run_llama_cli(length, type_param):
    """Execute llama-cli with the specified parameters"""
    
    # Command parameters
    model_path = "/root/zhenwei/data/models/qwen2-7b-instruct-q4_0.gguf"
    input_file = f"/root/zhenwei/data/{length}.txt"
    prompt_cache_file = f"/root/zhenwei/data/{length}_aligned.bin"
    thread_num = "222"
    tg = "10"

    if type_param == "GDS":
        bin_name = "build-gds/bin/llama-cli"
    else:
        bin_name = "build/bin/llama-cli"
    
    # Set context size based on length
    if length == "8192":
        context_size = "8300"
    else:  # 1024 or 4096
        context_size = "4300"
    
    # Build the command
    cmd = [
        "time",
        f"./{bin_name}",
        "-m", model_path,
        "-n", tg,           # number of tokens to generate
        "-c", context_size,   # context size (4300 for 1024/4096, 8300 for 8192)
        "-t", thread_num,          # number of threads
        "-f", input_file,     # input file
        "-ngl", "999",        # number of layers to offload to GPU
        "-e",                 # echo prompt
        "--temp", "0",        # temperature
        "-no-cnv",            # no conversation mode
        "-sm", "none"         # sampling method
    ]
    
    # Add prompt cache parameter if not prefill
    if type_param != "prefill":
        cmd.extend(["--prompt-cache", prompt_cache_file])
    
    print(f"Executing command with type={type_param}:")
    print(f"Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Prompt length: {length}")
    print(f"  Input file: {input_file}")
    print(f"  Text generation: {tg}")
    print(f"  Threads num: {thread_num}")
    print(f"  Context size: {context_size}")
    print()
    print("cmd is: ", " ".join(cmd))
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, 
                              capture_output=True,  # Capture output to parse time
                              text=True,
                              cwd="/root/zhenwei/llama.cpp")  # Set working directory
        
        # Parse time output from stderr
        if result.stderr:
            time_lines = result.stderr.strip().split('\n')
            
            if type_param != "prefill":
                for line in time_lines:
                    if 'load session time:' in line:
                        # Extract from "main: load session time: 98.786000 microseconds"
                        parts = line.split(':')
                        if len(parts) > 2:
                            time_part = parts[2].strip().split()[0]  # Get "98.786000"
                            try:
                                load_session_ms = float(time_part)  
                                print(f"Prompt processing time: {load_session_ms:.0f} ms")
                            except ValueError:
                                pass
                        break
            else:
                for line in time_lines:
                    if 'prompt eval time' in line:
                        # Extract from "llama_perf_context_print: prompt eval time =    1252.01 ms /  1024 tokens"
                        parts = line.split('=')
                        if len(parts) > 1:
                            time_part = parts[1].strip().split()[0]  # Get "1252.01"
                            try:
                                prompt_eval_ms = float(time_part)
                                print(f"Prompt processing time: {prompt_eval_ms:.0f} ms")
                            except ValueError:
                                pass
                        break
            
            # Extract total execution time
            for line in time_lines:
                if 'elapsed' in line:
                    # Extract time from "8.03user 3.80system 0:12.02elapsed 98%CPU" format
                    parts = line.split()
                    for part in parts:
                        if 'elapsed' in part:
                            time_part = part.replace('elapsed', '')
                            if ':' in time_part:
                                # Format: 0:12.02
                                minutes_part, seconds_part = time_part.split(':')
                                minutes = float(minutes_part)
                                seconds = float(seconds_part)
                                total_ms = (minutes * 60 + seconds) * 1000
                                print(f"Total execution time: {total_ms:.0f} ms")
                            else:
                                # Format: 12.02
                                seconds = float(time_part)
                                total_ms = seconds * 1000
                                print(f"Total execution time: {total_ms:.0f} ms")
                            break
                    break
                
    except FileNotFoundError:
        print("Error: llama-cli not found. Make sure you're in the correct directory.")
        print("Expected location: /root/zhenwei/llama.cpp/llama-cli")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_arguments()
    run_llama_cli(args.length, args.type)


