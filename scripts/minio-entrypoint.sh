#!/bin/sh
set -e

# Start MinIO with provided arguments
minio "$@" &
MINIO_PID=$!

# Wait until MinIO is ready to accept connections
echo "Waiting for MinIO to be ready..."
until curl -sf http://localhost:9000/minio/health/live > /dev/null; do
  sleep 1
done

echo "Configuring MinIO buckets..."
/usr/bin/mc alias set local http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

# Create the specified bucket
/usr/bin/mc mb --ignore-existing local/"$MINIO_BUCKET_NAME"

# Set the bucket to be not publicly accessible
/usr/bin/mc anonymous set none local/"$MINIO_BUCKET_NAME"
echo "MinIO is ready to accept connections."

# Wait for MinIO process
wait "$MINIO_PID"
