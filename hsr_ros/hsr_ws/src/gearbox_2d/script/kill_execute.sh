ps aux | grep execute_plan | grep -v grep | awk '{ print "kill -9", $2 }' | sh
