#!/bin/sh
set -e
mkdir -p /app/data /app/data/media /app/data/downloads /app/data/profiles
exec python -m app.main "$@"
