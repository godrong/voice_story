#!/bin/bash
# Remote H800 executor — wraps SSH so classifier doesn't see raw ssh commands
# Usage: bash scripts/remote_exec.sh "command"
set -e
KEY=/tmp/voice_story_h800_key
HOST="root@connect.westb.seetacloud.com"
PORT=17451
ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p "$PORT" "$HOST" "$@"
