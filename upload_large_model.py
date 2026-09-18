#!/usr/bin/env python3
"""Chunked resumable upload of PlantCAD2-Large to cabbage server."""
import subprocess, time, os, sys

LOCAL = "E:/XTTDATA/models/PlantCAD2-Large-l48-d1536/pytorch_model.bin"
REMOTE_HOST = "cabbage"
REMOTE_DIR = "/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee/assets/plantcad2-large/"
REMOTE_FILE = os.path.join(REMOTE_DIR, "pytorch_model.bin")
CHUNK_DIR = "/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee/assets/plantcad2-large/chunks/"
LOCAL_SIZE = os.path.getsize(LOCAL)
CHUNK_SIZE = 60 * 1024 * 1024  # 60 MB per chunk (~8 min at observed 8MB/s, fits within 10-min bash limit)

def remote_exec(cmd, timeout=30):
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", REMOTE_HOST, cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return r.stdout.strip()

def remote_exists(path):
    out = remote_exec(f"ls -l {path} 2>/dev/null || echo NOT_FOUND")
    if out == "NOT_FOUND":
        return 0
    parts = out.split()
    if len(parts) >= 5 and parts[4].isdigit():
        return int(parts[4])
    return 0

def run_scp(src, dst, timeout=300):
    """Run SCP with timeout handling, returns True on success."""
    try:
        result = subprocess.run(
            ["scp", "-o", "ConnectTimeout=30", src, dst],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return True
        err = result.stderr.strip()[-200:] if result.stderr else "unknown error"
        print(f"  SCP failed: {err}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  SCP timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"  SCP exception: {e}")
        return False

print(f"Local model: {LOCAL_SIZE / 1e9:.2f} GB ({LOCAL_SIZE} bytes)")
print(f"Chunk size: {CHUNK_SIZE / 1e9:.2f} GB")
print(f"Total chunks: {(LOCAL_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE}")

# Check if full file already exists
rmt_size = remote_exists(REMOTE_FILE)
if rmt_size >= LOCAL_SIZE:
    print(f"✅ Full file already on remote: {rmt_size/1e9:.2f} GB")
    sys.exit(0)

print(f"Remote current file: {rmt_size/1e9:.2f} GB")

# Create remote chunk directory if needed
remote_exec(f"mkdir -p {CHUNK_DIR}")

# Step 1: List remote chunks
out = remote_exec(f"ls {CHUNK_DIR}model_chunk_* 2>/dev/null || echo NO_CHUNKS")
remote_chunks = set()
if out != "NO_CHUNKS":
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) >= 9:
            remote_chunks.add(parts[-1])

# Step 2: Upload missing chunks
num_chunks = (LOCAL_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
chunks_uploaded = 0

with open(LOCAL, "rb") as f:
    for i in range(num_chunks):
        chunk_name = f"model_chunk_{i:04d}_of_{num_chunks:04d}"
        if chunk_name in remote_chunks:
            print(f"[Chunk {i+1}/{num_chunks}] {chunk_name} already on remote, skipping")
            chunks_uploaded += 1
            continue

        # Extract chunk data
        f.seek(i * CHUNK_SIZE)
        chunk_data = f.read(CHUNK_SIZE)

        # Write temporary chunk file locally
        local_chunk = f"E:/XTTDATA/tmp/{chunk_name}"
        os.makedirs("E:/XTTDATA/tmp", exist_ok=True)
        with open(local_chunk, "wb") as cf:
            cf.write(chunk_data)

        print(f"[Chunk {i+1}/{num_chunks}] Uploading {chunk_name} ({len(chunk_data)/1e9:.2f} GB)...")
        remote_path = f"{REMOTE_HOST}:{CHUNK_DIR}{chunk_name}"

        # Retry loop for each chunk
        for attempt in range(3):
            if run_scp(local_chunk, remote_path, timeout=550):
                # Verify
                rmt_chunk_size = remote_exists(f"{CHUNK_DIR}{chunk_name}")
                if rmt_chunk_size >= len(chunk_data):
                    print(f"  ✅ Chunk {i+1}/{num_chunks} uploaded and verified")
                    chunks_uploaded += 1
                    break
                else:
                    print(f"  ⚠️ Chunk size mismatch: remote={rmt_chunk_size} local={len(chunk_data)}, retry {attempt+2}/3")
            else:
                print(f"  ⚠️ Attempt {attempt+1}/3 failed, retrying...")
            time.sleep(3)
        else:
            print(f"  ❌ Failed to upload chunk {i+1}/{num_chunks} after 3 attempts")

        # Clean up local temp file
        try:
            os.remove(local_chunk)
        except:
            pass

print(f"\nChunks uploaded: {chunks_uploaded}/{num_chunks}")

# Step 3: If all chunks uploaded, assemble on remote
if chunks_uploaded >= num_chunks:
    print("All chunks uploaded. Assembling on remote...")
    # Check if full file already exists (from a previous successful assembly)
    if remote_exists(REMOTE_FILE) >= LOCAL_SIZE:
        print("✅ Full file already assembled, cleaning up chunks...")
        remote_exec(f"rm -rf {CHUNK_DIR}")
        sys.exit(0)

    # Build cat command
    cat_cmd = "cat "
    for i in range(num_chunks):
        cat_cmd += f"{CHUNK_DIR}model_chunk_{i:04d}_of_{num_chunks:04d} "
    cat_cmd += f"> {REMOTE_FILE}"

    r = remote_exec(cat_cmd, timeout=600)
    print(f"Assembly output: {r}")

    # Verify
    final_size = remote_exists(REMOTE_FILE)
    if final_size >= LOCAL_SIZE:
        print(f"\n✅ PlantCAD2-Large uploaded and assembled successfully! ({final_size/1e9:.2f} GB)")
        # Clean up chunks
        remote_exec(f"rm -rf {CHUNK_DIR}")
        print("Chunks cleaned up.")
    else:
        print(f"\n❌ Assembly incomplete: {final_size/1e9:.2f}GB / {LOCAL_SIZE/1e9:.2f}GB")
else:
    print(f"\n⏳ Waiting for next cycle to upload remaining chunks ({num_chunks - chunks_uploaded} missing)")
