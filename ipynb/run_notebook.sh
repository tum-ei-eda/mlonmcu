#!/bin/bash

set -e

SHORT=h:,:,e:
LONG=home:,environment:,skip,cleanup,clear,noop,html,pdf,help
OPTS=$(getopt -a -n class --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

HOME_=
VENV=
SKIP=0
CLEANUP=0
CLEAR=0
NOOP=0
HTML=0
PDF=0

function print_usage() {
    echo "Usage: $0 path/to/notebook.ipynb [-h MLONMCU_HOME] [-e VENV] [--skip] [--cleanup] [--clear] [--noop] [--html] [--pdf]"
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
    --html )
      HTML=1
      shift
      ;;
    --pdf )
      PDF=1
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
TEMPDIR=$(mktemp -d -t $NAME-XXXX)
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
        echo "Setting up MLonMCU environment..."
        python3 -m mlonmcu.cli.main setup -v
    fi
fi

if [[ $NOOP -eq 0 ]]
then
    echo "Executing notebook..."
    python3 -m jupyter nbconvert --to notebook --execute $NOTEBOOK --inplace
else
    echo "Skipping execution of notebook..."
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

cd -

if [[ $CLEANUP -eq 1 ]]
then
    echo "Cleaning up..."
    rm -rf $TEMPDIR
fi

echo "Done."
