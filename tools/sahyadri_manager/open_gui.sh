#!/bin/bash
# On the Mac: tunnel the manager's page (a Unix socket on pawna readable only by you) to http://localhost:PORT and
# open it in the browser. Ctrl-C closes the tunnel.   usage: open_gui.sh [port, default 8765]
PORT=${1:-8765}
HOST=${SAHYADRI_HOST:-shadab@pawna.theory.tifr.res.in}
RSOCK=${SAHYADRI_SOCK:-/user/shadab/.sahyadri_mgr/gui.sock}
echo "tunnel localhost:$PORT -> $HOST:$RSOCK (Ctrl-C to close)"
( sleep 2; open "http://localhost:$PORT/" 2>/dev/null || xdg-open "http://localhost:$PORT/" ) &
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -L "$PORT:$RSOCK" "$HOST"
