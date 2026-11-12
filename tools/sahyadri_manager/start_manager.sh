#!/bin/bash
# Start (or show) the Sahyadri run manager on the pawna head node, in a detached screen session "sahyadri_mgr".
#   start_manager.sh          start if not running
#   start_manager.sh stop     stop it (jobs and runs are not affected)
#   start_manager.sh status
# Restart after a reboot: add to `crontab -e`:  @reboot <this repo>/tools/sahyadri_manager/start_manager.sh
HERE=$(dirname "$(readlink -f "$0")")
S=sahyadri_mgr
running() { screen -ls 2>/dev/null | grep -q "[.]$S[[:space:]]"; }
case "${1:-start}" in
  stop)   running && screen -S $S -X stuff $'\003' && sleep 2; running && screen -S $S -X quit; echo stopped ;;
  status) running && echo "running (screen -r $S to look, Ctrl-a d to leave)" || echo "not running"
          tail -5 ~/.sahyadri_mgr/manager.log 2>/dev/null ;;
  start)  if running; then echo "already running (screen -r $S)"; exit 0; fi
          command -v inotifywait > /dev/null || { echo "inotifywait not found (inotify-tools)"; exit 1; }
          screen -dmS $S bash -c "cd '$HERE' && exec /usr/bin/python3 manager.py serve >> ~/.sahyadri_mgr/screen.log 2>&1"
          sleep 2; running && echo "started: screen -r $S; page socket ~/.sahyadri_mgr/gui.sock" || { echo "did not start"; tail ~/.sahyadri_mgr/screen.log; exit 1; } ;;
esac
