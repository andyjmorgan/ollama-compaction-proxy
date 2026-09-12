#!/usr/bin/env bash
# Cross-compile for the Spark (linux/arm64) and install as a systemd daemon
# beside Ollama and the tokenizer service.
#
#   ./deploy/deploy.sh [host]
set -euo pipefail

HOST="${1:-192.168.69.28}"
BIN=ollama-compaction-proxy
UNIT=ollama-compaction-proxy.service

cd "$(dirname "$0")/.."

echo "==> building ${BIN} for linux/arm64"
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath \
    -ldflags="-s -w" -o "/tmp/${BIN}" ./cmd/server

echo "==> copying to ${HOST}"
scp -q "/tmp/${BIN}" "deploy/${UNIT}" "deploy/claude-models.json" "${HOST}:/tmp/"

echo "==> installing"
ssh "${HOST}" "
    set -euo pipefail
    sudo mkdir -p /etc/ollama-compaction-proxy
    sudo install -m 0644 /tmp/claude-models.json /etc/ollama-compaction-proxy/claude-models.json
    sudo install -m 0755 /tmp/${BIN} /usr/local/bin/${BIN}
    sudo install -m 0644 /tmp/${UNIT} /etc/systemd/system/${UNIT}

    # Persistent HMAC key: generate once, keep across deploys.
    if ! sudo test -f /etc/ollama-compaction-proxy/env; then
        sudo mkdir -p /etc/ollama-compaction-proxy
        echo \"COMPACT_HMAC_KEY=\$(openssl rand -hex 32)\" | sudo tee /etc/ollama-compaction-proxy/env >/dev/null
        sudo chmod 0600 /etc/ollama-compaction-proxy/env
        sudo chown ollama:ollama /etc/ollama-compaction-proxy/env
    fi

    sudo systemctl daemon-reload
    sudo systemctl enable --now ${UNIT}
    sudo systemctl restart ${UNIT}
    rm -f /tmp/${BIN} /tmp/${UNIT} /tmp/claude-models.json
"

echo "==> waiting for health"
for _ in $(seq 1 20); do
    if curl -fsS --max-time 2 "http://${HOST}:8082/health" >/dev/null 2>&1; then
        echo "healthy: http://${HOST}:8082"
        curl -fsS --max-time 2 "http://${HOST}:8082/health"; echo
        exit 0
    fi
    sleep 0.5
done

echo "service did not become healthy; recent logs:" >&2
ssh "${HOST}" "sudo journalctl -u ${UNIT} -n 30 --no-pager" >&2
exit 1
