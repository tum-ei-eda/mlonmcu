#!/bin/bash

function monitor() {
    # DEST=$1
    INTERVAL=5
    /usr/bin/time -f "Elapsed:%E, CPU: %P" "$@" &
    myPid=$!
    OUT=./$myPid.metrics.csv
    echo "TS;MEM;CPU;DISK;1M;5M;10M" > $OUT
    while kill -0 "$myPid"
    do
        TS=$(date +%s)
        MEM=$(free | grep Mem | tr -s ' ' |  cut -d' ' -f3)
        CPU=$(ps -A -o pcpu | tail -n+2 | paste -sd+ | bc)
        # DISK=$(df . | tail -1 | cut -d' ' -f3)
        DISK=$(df /data | tail -1 | cut -d' ' -f3)
        UPTIME=$(uptime | grep -oP '(?<=average:).*' | tr -d '[:blank:]' | tr ',' ';')
        echo "$TS;$MEM;$CPU;$DISK;$UPTIME" >> $OUT
        sleep $INTERVAL
    done
    echo "Monitoring Results:"
    cat $OUT
    MAX_MEM=$(cat $OUT | tail -n +2 | awk -F';' 'BEGIN{a=0}{if ($2>(0+a)) a=$2 fi}END{print a}')
    MIN_MEM=$(cat $OUT | tail -n +2 | awk -F';' 'BEGIN{a=99999999999}{if ($2<(0+a)) a=$2 fi}END{print a}')
    AVG_MEM=$(cat $OUT | tail -n +2 | awk -F ';' '{ sum += $2 } END { if (NR > 0) print sum / NR }')
    MAX_CPU=$(cat $OUT | tail -n +2 | awk -F';' 'BEGIN{a=0}{if ($3>(0+a)) a=$3 fi}END{print a}')
    MIN_CPU=$(cat $OUT | tail -n +2 | awk -F';' 'BEGIN{a=9999999999}{if ($3<(0+a)) a=$3 fi}END{print a}')
    AVG_CPU=$(cat $OUT | tail -n +2 | awk -F ';' '{ sum += $3 } END { if (NR > 0) print sum / NR }')
    MAX_DISK=$(cat $OUT | tail -n +2 | awk -F';' 'BEGIN{a=0}{if ($4>(0+a)) a=$4 fi}END{print a}')
    MIN_DISK=$(cat $OUT | tail -n +2 | awk -F';' 'BEGIN{a=9999999999}{if ($4<a) a=$4 fi}END{print a}')
    MAX_1M=$(cat $OUT | tail -n +2 | awk -F';' 'BEGIN{a=0}{if ($5>(0+a)) a=$5 fi}END{print a}')
    MAX_5M=$(cat $OUT | tail -n +2 | awk -F';' 'BEGIN{a=0}{if ($6>(0+a)) a=$6 fi}END{print a}')
    MAX_10M=$(cat $OUT | tail -n +2 | awk -F';' 'BEGIN{a=0}{if ($7>(0+a)) a=$7 fi}END{print a}')
    echo "Max. MEM: $MAX_MEM"
    echo "Min. MEM: $MIN_MEM"
    echo "Avg. MEM: $AVG_MEM"
    echo "Max. CPU: $MAX_CPU"
    echo "Min. CPU: $MIN_CPU"
    echo "Avg. CPU: $AVG_CPU"
    echo "Max. DISK: $MAX_DISK"
    echo "Min. DISK: $MIN_DISK"
    echo "Max. 1M: $MAX_1M"
    echo "Max. 5M: $MAX_5M"
    echo "Max. 10M: $MAX_10M"

}

function run_mlonmcu() {
    EXECUTOR=$1
    P=$2
    N=$3
    monitor time python3 -m mlonmcu.cli.main flow run aww --target etiss --backend tvmaot --platform mlif --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ -c session.use_init_stage=0 -c runs_per_stage=0 --progress --parallel $P -c mlif.num_threads=$N -c tvmaot.num_threads=$N -c session.executor=$EXECUTOR
    # monitor python3 -m mlonmcu.cli.main flow run aww --target etiss --backend tvmaot --platform mlif --config-gen _ --config-gen _ -c session.use_init_stage=0 -c runs_per_stage=0 --progress --parallel $P -c mlif.num_threads=$N -c tvmaot.num_threads=$N -c session.executor=$EXECUTOR

}

# run_mlonmcu process_pool 1 1
# run_mlonmcu process_pool 1 2
# run_mlonmcu process_pool 1 4
# run_mlonmcu process_pool 1 8
# run_mlonmcu process_pool 2 1
# run_mlonmcu process_pool 2 2
# run_mlonmcu process_pool 2 4
# run_mlonmcu process_pool 2 8
# run_mlonmcu process_pool 4 1
# run_mlonmcu process_pool 4 2
# run_mlonmcu process_pool 4 4
# run_mlonmcu process_pool 4 8
# run_mlonmcu process_pool 8 1
# run_mlonmcu process_pool 8 2
# run_mlonmcu process_pool 8 4
run_mlonmcu process_pool 8 8
