#!/bin/bash

myPid="none"

control_c() {
    echo "Handling keyboardinterrupt!"
    if [[ "$myPid" != "none" ]]
    then
        echo "Killing $myPid"
        # kill -9 $myPid
        pkill -TERM -P $myPid
        exit
    fi
}

trap control_c SIGINT

function monitor() {
    # DEST=$1
    INTERVAL=5
    /usr/bin/time -f "Elapsed:%E, CPU: %P" "$@" &
    myPid=$!
    OUT=./$myPid.metrics.csv
    echo "TS;MEM;CPU;DISK;1M;5M;10M" > $OUT
    while kill -0 "$myPid" 2> /dev/null
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
    P_=0  # R?
    N=$3
    B=1
    MULTIPLIER=$4
    shift 4
    ARGS=$@
    RPC_TRACKER=gpu2.eda.cit.tum.de:9000
    RPC_KEY=default
    if [[ "$EXECUTOR" == "rpc" ]]
    then
        P_=$P
        RPC_ARGS="-c session.rpc_tracker=$RPC_TRACKER -c session.rpc_key=$RPC_KEY -c session.parallel_jobs=$P_"
        P=$(nproc)
        B=$P_
    fi

    SESSION_ARGS="-c session.use_init_stage=0 -c print_report=0 -c runs_per_stage=0 --parallel $P -c mlif.num_threads=$N -c tvmaot.num_threads=$N -c session.executor=$EXECUTOR -c session.batch_size=$B --progress -c run.export_optional=1"
    HOST=$(hostname -f)
    CMD="python3 -m mlonmcu.cli.main $ARGS $SESSION_ARGS $RPC_ARGS"
    CORES=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')
    THREADS=$(grep -c ^processor /proc/cpuinfo)
    RAM=$(grep MemTotal /proc/meminfo | awk '{print $2 / 1024 / 1024, "GiB"}')
    for i in $(eval echo "{1..$MULTIPLIER}")
    do
        CMD="$CMD --config-gen _"
    done
    echo "======================"
    echo "Host: $HOST (${CORES}C${THREADS}T) RAM: $RAM"
    echo "Scenario: EXECUTOR=$EXECUTOR MULTIPLIER=$MULTIPLIER ARGS=$ARGS"
    echo "Config: P=$P P_=$P_ B=$B N=$N"
    echo "> $CMD"
    monitor $CMD
    echo "Cooldown..."
    sleep 30
    echo "----------------------"

    # --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _ --config-gen _
    # monitor python3 -m mlonmcu.cli.main flow run aww --target etiss --backend tvmaot --platform mlif --config-gen _ --config-gen _ -c session.use_init_stage=0 -c runs_per_stage=0 --progress --parallel $P -c mlif.num_threads=$N -c tvmaot.num_threads=$N -c session.executor=$EXECUTOR

}

# Framework: TVM
# run_mlonmcu process_pool 1 1 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 1 2 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 1 4 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 1 8 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 2 1 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 2 2 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 2 4 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 2 8 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 4 1 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 4 2 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 4 4 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 4 8 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 8 1 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 8 2 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 8 4 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu process_pool 8 8 64 flow run aww --target etiss --backend tvmaot --platform mlif

# run_mlonmcu rpc 1 1 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 1 2 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 1 4 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 1 8 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 2 1 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 2 2 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 2 4 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 2 8 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 4 1 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 4 2 64 flow run aww --target etiss --backend tvmaot --platform mlif
run_mlonmcu rpc 4 4 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 4 8 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 8 1 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 8 2 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 8 4 64 flow run aww --target etiss --backend tvmaot --platform mlif
# run_mlonmcu rpc 8 8 64 flow run aww --target etiss --backend tvmaot --platform mlif

# Framework: TFLM
# run_mlonmcu process_pool 1 1 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 1 2 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 1 4 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 1 8 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 1 16 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 1 32 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 2 1 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 2 2 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 2 4 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 2 8 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 2 16 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 2 32 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 4 1 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 4 2 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 4 4 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 4 8 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 4 16 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 4 32 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 8 1 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 8 2 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 8 4 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 8 8 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 8 16 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 8 32 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 16 1 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 16 2 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 16 4 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 16 8 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 16 16 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 16 32 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 32 1 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 32 2 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 32 4 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 32 8 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 32 16 64 flow run aww --target etiss --backend tflmi --platform mlif
# run_mlonmcu process_pool 32 32 64 flow run aww --target etiss --backend tflmi --platform mlif
