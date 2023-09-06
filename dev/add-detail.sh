#!/bin/bash

RED="\033[91m"
GREEN="\033[92m"
EOC="\033[0m"

git -C . rev-parse
if [ $? -ne 0 ]; then
    echo -e $RED"ERROR: '$PWD' is not a git repo."$EOC
    echo "Run this script in the dir of a git repo."
    exit 1
fi

HELP=0
for i in "$@"
do
    case "$i" in
      -h | --help)
        HELP=1
        ;;
    esac
done

if [ $HELP -eq 1 ]; then
    echo Usage: add-detail.sh [changetype] [summary] [description] [options]
    echo
    echo This script calls 'detail' and a sequence of git comannds to automate the work of
    echo adding a note to the latest commit. For more information about 'detail', refer to
    echo its document: https://detail.readthedocs.io
    echo
    echo Positional arguments:
    echo " changetype   Type of the change: trivial|feature|bug|api-break. (default: trivial)"
    echo " summary      One line summary of the change. If not provided or an empty string is"
    echo "              specified, it will default to the subject (i.e., the first line) of"
    echo "              the commit message."
    echo " description  Detailed description of the change. Not needed for 'trivial' changes."
    echo "              This can be a multi-line text. If not provided or an empty string is"
    echo "              specified, it will default to the body of the commit message (i.e.,"
    echo "              the whole commit message minus the first line)."
    echo
    echo Options:
    echo " -h --help   Show this help message and quit."
    exit 0
fi

detail --help > /dev/null
if [ $? -ne 0 ]; then
    echo -e $RED"ERROR: 'detail' seems not installed."$EOC
    echo "You may do 'pip install detail' to install it. For more information, refer to"
    echo "  https://detail.readthedocs.io"
    exit 1
fi

git log -1 > /dev/null
if [ $? -ne 0 ]; then
    echo -e $RED"ERROR: There is no commit yet."$EOC
    exit 1
fi

changetype="$1"
summary="$2"
description="$3"
if [[ "$changetype" == "" ]]; then
    changetype="trivial"
fi
if [[ "$changetype" != "trivial" && "$changetype" != "feature" && "$changetype" != "bug" && "$changetype" != "api-break" ]]; then
    echo -e $RED"ERROR: changetype's value must be one of trivial, feature, bug, and api-break."$EOC
    exit 1
fi
if [[ "$summary" == "" ]]; then
    summary=$(git log -1 --pretty=%s)
fi
if [[ "$description" == "" ]]; then
    description=$(git log -1 --pretty=%b)
fi

function call_detail() {
    detail << EOF
$(echo -e "$1
$2
$3\e\r")
EOF
}

result=$(call_detail "$changetype" "$summary" "$description")
note_fname=$(echo $result | rev | cut -f 1 -d ' ' | rev)
git add "$note_fname"
git commit --amend --no-edit
echo -e "git show --name-only HEAD\n"
git show --name-only HEAD
echo
echo -e $GREEN"detail note successfully added."$EOC
