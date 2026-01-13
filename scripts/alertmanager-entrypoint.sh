#!/bin/sh
# Alertmanager entrypoint script to substitute environment variables

set -e

# Check if gettext (envsubst) is available
if ! command -v envsubst > /dev/null 2>&1; then
    echo "Warning: envsubst not found. Installing gettext..."
    apk add --no-cache gettext 2>/dev/null || {
        echo "Error: Could not install envsubst. Using config file as-is."
        cp /etc/alertmanager/alertmanager.yml /tmp/alertmanager.yml
    }
fi

# Substitute environment variables in the config file
if command -v envsubst > /dev/null 2>&1; then
    envsubst < /etc/alertmanager/alertmanager.yml > /tmp/alertmanager.yml
else
    cp /etc/alertmanager/alertmanager.yml /tmp/alertmanager.yml
fi

# Start Alertmanager with the processed config
exec /bin/alertmanager \
  --config.file=/tmp/alertmanager.yml \
  --storage.path=/alertmanager \
  --web.external-url=http://localhost:9093 \
  "$@"
