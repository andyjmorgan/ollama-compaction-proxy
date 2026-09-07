#!/usr/bin/env bash
# Cross-compile for the Spark (linux/arm64) and install as a systemd daemon
# alongside Ollama.
#
#   ./deploy/deploy.sh [host]
#
# Defaults to office-spark-1. The binary is pure Go, so no toolchain is needed
# on the target.
set -euo pipefail

HOST="${1:-192.168.69.28}"
BIN=ollama-tokenizer
UNIT=ollama-tokenizer.service

cd "$(dirname "$0")/.."

echo "==> building ${BIN} for linux/arm64"
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath \
    -ldflags="-s -w" -o "/tmp/${BIN}" ./cmd/server

echo "==> copying to ${HOST}"
scp -q "/tmp/${BIN}" "deploy/${UNIT}" "${HOST}:/tmp/"

echo "==> installing"
ssh "${HOST}" "
    set -euo pipefail
    sudo install -m 0755 /tmp/${BIN} /usr/local/bin/${BIN}
    sudo install -m 0644 /tmp/${UNIT} /etc/systemd/system/${UNIT}
    sudo systemctl daemon-reload
    sudo systemctl enable --now ${UNIT}
    sudo systemctl restart ${UNIT}
    rm -f /tmp/${BIN} /tmp/${UNIT}
"

echo "==> waiting for health"
for _ in $(seq 1 20); do
    if curl -fsS --max-time 2 "http://${HOST}:8081/health" >/dev/null 2>&1; then
        echo "healthy: http://${HOST}:8081"
        exit 0
    fi
    sleep 0.5
done

echo "service did not become healthy; recent logs:" >&2
ssh "${HOST}" "sudo journalctl -u ${UNIT} -n 30 --no-pager" >&2
exit 1
