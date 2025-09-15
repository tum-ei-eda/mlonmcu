#!/bin/bash

set -eE

SHORT=h:,:,e:
LONG=home:,environment:,skip,cleanup,clear,noop,html,pdf,mill,dump,help
OPTS=$(getopt -a -n class --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

HOME_=
VENV=
SKIP=0
CLEANUP=0
CLEAR=0
NOOP=0
FAIL=0
HTML=0
PDF=0
MILL=0
DUMP=0

function print_usage() {
    echo "Usage: $0 path/to/notebook.ipynb [-h MLONMCU_HOME] [-e VENV] [--skip] [--cleanup] [--clear] [--noop] [--html] [--pdf] [--mill] [--dump]"
}

while :
do
  case "$1" in
    -h | --home )
      HOME_="$2"
      shift 2
      ;;
    -e | --environment )
      VENV="$2"
      shift 2
      ;;
    --skip )
      SKIP=1
      shift
      ;;
    --cleanup )
      CLEANUP=1
      shift
      ;;
    --clear )
      CLEAR=1
      shift
      ;;
    --noop )
      NOOP=1
      shift
      ;;
    --fail )
      FAIL=1
      shift
      ;;
    --html )
      HTML=1
      shift
      ;;
    --pdf )
      PDF=1
      shift
      ;;
    --mill )
      MILL=1
      shift
      ;;
    --dump )
      DUMP=1
      shift
      ;;
    --help)
      print_usage
      exit 0
      ;;
    --)
      if [[ "$NOTEBOOK" != "" ]]
      then
          echo "Too many notebooks specified!"
          exit 1
      fi
      NOTEBOOK=$2
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      print_usage
      exit 1
      ;;
  esac
done


if [[ -z $NOTEBOOK ]]
then
    echo "No NOTEBOOK specified. Aborting..."
    exit 1
fi
NOTEBOOK=$(readlink -f $NOTEBOOK)
if [[ ! -f $NOTEBOOK ]]
then
    echo "Notebook does not exist!"
    exit 1
fi
DIR=$(readlink -f $(dirname $0))
NAME=$(basename $NOTEBOOK | cut -d. -f1)
DIRECTORY=$(dirname $NOTEBOOK)
YAML=$DIRECTORY/environment.yml.j2
REQUIREMENTS=$DIRECTORY/requirements.txt
PLUGINS_DIR=$DIRECTORY/plugins
TEMPDIR=$(mktemp -d -t $NAME-XXXX)

function __error_handing__(){
    local last_status_code=$1;
    local error_line_number=$2;
    echo 1>&2 "Error - exited with status $last_status_code at line $error_line_number";
    if [[ $CLEANUP -eq 1 ]]
    then
        echo "Cleaning up after failure..."
        rm -rf $TEMPDIR
    fi
}
trap  '__error_handing__ $? $LINENO' ERR

if [[ "$HOME_" != "" ]]
then
    WORKSPACE=$HOME_
else
    WORKSPACE=$TEMPDIR/workspace
fi
if [[ "$VENV" != "" ]]
then
    ENVIRONMENT=$VENV
else
    ENVIRONMENT=$TEMPDIR/venv
fi
cd $TEMPDIR

if [[ "$ENVIRONMENT" != "custom" ]]
then
    echo "Configuring python..."
    # export PYTHONPATH=$DIR/.. # TODO
    if [[ $SKIP -eq 0 ]]
    then
        echo "Creating virtualenv..."
        python3 -m venv $ENVIRONMENT
        source $ENVIRONMENT/bin/activate
        python3 -m pip install -e $DIR/..
        if [[ -f $REQUIREMENTS ]]
        then
            echo "(Using provided requirements)"
            python3 -m pip install -r $REQUIREMENTS > /dev/null
        fi
        elif [[ -f $DIR/requirements.txt ]]
        then
            echo "(No requirements found. Falling back to default requirements.txt)"
            python3 -m pip install -r $DIR/requirements.txt > /dev/null
    else
        echo "Skipping creation of virtualenv..."
    fi
fi
echo "PYTHONPATH=$PYTHONPATH"
if [[ "$WORKSPACE" != "custom" ]]
then
    export MLONMCU_HOME=$WORKSPACE
    if [[ $SKIP -eq 0 ]]
    then
        echo "Initializing MLonMCU environment..."
        if [[ -f $YAML ]]
        then
            echo "(Using provided template)"
            python3 -m mlonmcu.cli.main init -t $YAML $WORKSPACE --non-interactive --clone-models --allow-exists
        else
            echo "(No template found. Falling back to default...)"
            python3 -m mlonmcu.cli.main init -t default $WORKSPACE --non-interactive --clone-models --allow-exists
        fi
        if [[ -d $PLUGINS_DIR ]]
        then
            echo "Adding plugins to workspace"
            cp -r $PLUGINS_DIR/* $WORKSPACE/plugins/
        fi
        echo "Setting up MLonMCU environment..."
        python3 -m mlonmcu.cli.main setup -v
    fi
fi

FAILING=0
if [[ $NOOP -eq 0 ]]
then
    echo "Executing notebook..."
    cp $NOTEBOOK $NOTEBOOK.orig
    if [[ $MILL -eq 1 ]]
    then
        python3 -m pip install papermill
        python3 -m papermill $NOTEBOOK $NOTEBOOK --cwd $DIRECTORY 2>&1 | tee $TEMPDIR/out.txt
        EXIT=$?
    else
        python3 -m jupyter nbconvert --to notebook --execute $NOTEBOOK --inplace 2>&1 | tee $TEMPDIR/out.txt
        EXIT=$?
    fi
    if [[ $EXIT -eq 0 ]]
    then
        (cat out.txt | grep -q "Traceback") && FAILING=1 || FAILING=0
    else
        FAILING=1
    fi
else
    echo "Skipping execution of notebook..."
fi

if [[ $DUMP -eq 1 ]]
then
   echo "Dumping environment.yml.j2..."
   cat $WORKSPACE/environment.yml
   echo
   echo "Dumping python packages..."
   python3 -m pip freeze
   echo
fi

if [[ $HTML -eq 1 ]]
then
    echo "Converting notebook to HTML..."
    python3 -m jupyter nbconvert --to html $NOTEBOOK
fi

if [[ $PDF -eq 1 ]]
then
    echo "Converting notebook to PDF..."
    python3 -m jupyter nbconvert --to pdf $NOTEBOOK
fi

if [[ $CLEAR -eq 1 ]]
then
    echo "Clearing output cells..."
    python3 -m jupyter nbconvert --clear-output --inplace $NOTEBOOK
fi

if [[ $FAILING -eq 1 ]]
then
    echo "Errors occured during notebook execution..."
    echo "Outputs:"
    cat $TEMPDIR/out.txt
    if [[ $FAIL -eq 1 ]]
    then
        echo "Aborting:"
        exit 1
    fi
fi

cd -

if [[ $CLEANUP -eq 1 ]]
then
    echo "Cleaning up..."
    df
    rm -rf $TEMPDIR
    df
fi

echo "Done."
