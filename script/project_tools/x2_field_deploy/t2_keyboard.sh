#!/bin/bash
# t2_keyboard.sh  (SoC1, agi@) - keyboard intents -> kplanner :5563
# w=前进0.35m/s  s=后退0.1  a/d=左/右转0.65rad/s  space=停  q=退出(先归零)
exec /home/agi/gear-sonic/venv/bin/python /home/agi/walk/x2_keyboard_walk_publisher.py --mode planner-cmd
