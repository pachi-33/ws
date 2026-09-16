#!/usr/bin/env bash

# Print the Docker container name that owns a host PID.

set -u

usage() {
    echo "Usage: $0 <PID>" >&2
    exit 2
}

if [[ $# -ne 1 ]]; then
    usage
fi

pid=$1
if [[ ! "$pid" =~ ^[0-9]+$ ]] || [[ "$pid" -le 0 ]]; then
    echo "Error: PID must be a positive integer." >&2
    exit 2
fi

if [[ ! -d "/proc/$pid" ]]; then
    echo "Error: process $pid does not exist or is no longer running." >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker command was not found." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Error: cannot access the Docker daemon; try running with sudo or check Docker permissions." >&2
    exit 1
fi

# Fast and precise path: Docker normally records the full container ID in the
# process cgroup path, for example /docker/<id> or docker-<id>.scope.
while IFS= read -r container_id; do
    if container_name=$(docker inspect \
        --type container \
        --format '{{.Name}}' \
        "$container_id" 2>/dev/null); then
        printf '%s\n' "${container_name#/}"
        exit 0
    fi
done < <(grep -oE '[0-9a-f]{64}' "/proc/$pid/cgroup" 2>/dev/null | awk '!seen[$0]++')

# Fallback: compare PID namespaces. This also handles cgroup layouts that do
# not expose the Docker container ID directly.
target_namespace=$(readlink "/proc/$pid/ns/pid" 2>/dev/null || true)
matches=()

if [[ -n "$target_namespace" ]]; then
    while IFS= read -r container_id; do
        [[ -n "$container_id" ]] || continue

        init_pid=$(docker inspect \
            --format '{{.State.Pid}}' \
            "$container_id" 2>/dev/null || true)

        [[ "$init_pid" =~ ^[0-9]+$ ]] || continue
        [[ "$init_pid" -gt 0 ]] || continue

        container_namespace=$(readlink "/proc/$init_pid/ns/pid" 2>/dev/null || true)
        if [[ "$container_namespace" == "$target_namespace" ]]; then
            container_name=$(docker inspect \
                --format '{{.Name}}' \
                "$container_id" 2>/dev/null || true)
            [[ -n "$container_name" ]] && matches+=("${container_name#/}")
        fi
    done < <(docker ps -q)
fi

if [[ ${#matches[@]} -eq 1 ]]; then
    printf '%s\n' "${matches[0]}"
    exit 0
elif [[ ${#matches[@]} -gt 1 ]]; then
    # This can happen when containers deliberately share one PID namespace.
    printf '%s\n' "${matches[@]}"
    exit 0
fi

echo "No Docker container found for PID $pid; it may be a host process or belong to another container runtime." >&2
exit 1
