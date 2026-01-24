#!/bin/bash
# Monitor load balancer and restart if not running

LB_DIR="/home/ttanveer/algoforge/examples/eplb"
LOG="/tmp/load_balancer.log"

while true; do
    if ! pgrep -f "python.*load_balancer.py" > /dev/null; then
        echo "$(date): Load balancer not running, restarting..." >> /tmp/lb_monitor.log
        cd "$LB_DIR" && nohup python load_balancer.py >> "$LOG" 2>&1 &
    fi
    sleep 600  # Check every 10 minutes
done
