#!/usr/bin/env bash
set -euo pipefail
curl -s http://127.0.0.1:8000/v1/systemone   -H 'content-type: application/json'   -d @examples/request-mixed.json
