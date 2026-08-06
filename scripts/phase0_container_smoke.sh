#!/usr/bin/env bash
set -euo pipefail

id -u | grep -qx '65534'
grep -Eq '^CapEff:[[:space:]]+0+$' /proc/self/status
grep -Eq '^NoNewPrivs:[[:space:]]+1$' /proc/self/status

if find /sys/class/net -mindepth 1 -maxdepth 1 | grep -qv '/lo$'; then
    echo "unexpected network interface" >&2
    exit 12
fi

if touch /etc/phase0-write-test 2>/tmp/etc-write-error; then
    echo "root filesystem is writable" >&2
    exit 13
fi

if touch /models/phase0-write-test 2>/tmp/model-write-error; then
    echo "model mount is writable" >&2
    exit 14
fi

touch /tmp/tmpfs-write-ok
nvidia-smi \
    --query-gpu=name,uuid,memory.total,driver_version,compute_cap \
    --format=csv,noheader |
    tee /results/container-gpu-smoke.txt

printf '%s\n' \
    'uid=65534' \
    'cap_eff=0' \
    'no_new_privs=1' \
    'network_interfaces=lo' \
    'root_read_only=1' \
    'model_mount_read_only=1' \
    'results_mount_writable=1' \
    'tmpfs_writable=1' \
    > /results/container-isolation-smoke.txt
