#!/bin/bash

RED="\033[91m"
GREEN="\033[92m"
EOC="\033[0m"

ABORT=0
CONTINUE=0
HELP=0
for i in "$@"
do
    case "$i" in
      --abort)
        ABORT=1
        ;;
      --continue)
        CONTINUE=1
        ;;
      -h | --help)
        HELP=1
        ;;
    esac
done

if [ $HELP -eq 1 ]; then
    echo Usage: footing-auto.sh [options]
    echo
    echo This script calls 'footing update' and a sequence of git comannds to help user
    echo perform 'footing update' in a more automated fashing. It also features more user
    echo friendly error messages and useful hints to flatten the learning curve of the
    echo tools. Though in most cases you can spare yourself of concerning a lot of details
    echo in footing update with this script, all commands called by this script will
    echo be printed out to the stdout one by one as the execution proceeds, and so the
    echo entire procedure is completely transparent to the user.
    echo
    echo By the current implementation of this script, footing will be updated to the
    echo latest tag of the template. There is no option to select the revision of the
    echo template with this script. But user can run 'footing update -v <ref>' directly.
    echo
    echo When all commands are done successfully and there is no merge conflicts to be
    echo resolved manually. The footing will be updated auomatically.
    echo
    echo In cases there are conflicts, the workflow will stop. User should resolve them,
    echo and call 'git add <conflicts-resolved-files>', and then 'footing-auto.sh --continue'
    echo to finish the updating.
    echo
    echo In cases you want to abort the updating without resolving the conflicts, you
    echo can call 'footing-auto.sh --abort'. The current branch will be recovered to the
    echo one where footing-auto.sh was initially called, and the temporary '_footing_update'
    echo branch will be deleted.
    echo
    echo Options:
    echo " --abort     Abort the updating."
    echo " --continue  Continue the updating after resolving the merge conflicts."
    echo " -h --help   Show this help message and quit."
    exit 0
fi

git -C . rev-parse
if [ $? -ne 0 ]; then
    echo -e $RED"ERROR: '$PWD' is not a git repo."$EOC
    echo "Run this script in the dir of a git repo."
    exit 1
fi

template_latest_tag=$(git ls-remote -tq --refs --sort=-v:refname git@gitlab.com:drailab/drailab-lib-template.git | head -1 | cut -f 3 -d /)
cur_branch=$(git rev-parse --abbrev-ref HEAD)

if [ $ABORT -eq 1 ]; then
    if [[ "$cur_branch" != "_footing_update" ]]; then
        echo -e $RED"ERROR: The current branch is '$cur_branch', not '_footing_update'."$EOC
        echo Must be in '_footing_update' branch for this command to work.
        exit 1
    fi
    echo Aborting the footing update...
    # Gets the parent branch's name.
    parent=$(git show-branch -a | awk -F'[]^~[]' '/^\*/ && !/'"$cur_branch"'/ {print $2;exit}')
    set -ex
    git merge --abort || true
    git checkout $parent
    git branch -D _footing_update || true
    git branch -D _footing_update_temp || true
    set +x
    exit 0
fi

if [ $CONTINUE -eq 0 ]; then
    if [[ "$cur_branch" == "_footing_update" || "$cur_branch" == "_footing_update_temp" ]]; then
        echo -e $RED"ERROR: The current branch is '$cur_branch'."$EOC
        echo "You probably forgot to add the --continue option."
        echo "Or if you wanted to do 'footing update' from scratch, you must not be in a"
        echo "temporary branch ('_footing_update' or '_footing_update_temp') to call 'footing update'."
        echo "Please switch to a different branch, type 'git branch -D $cur_branch' to delete the"
        echo "temporary branch, and then try again."
        exit 1
    fi

    footing --version > /dev/null
    if [ $? -ne 0 ]; then
        echo -e $RED"ERROR: 'footing' seems not installed."$EOC
        echo "You may do 'pip install footing' to install it. For more information, refer"
        echo "  to https://pypi.org/project/footing/"
        exit 1
    fi
    echo + footing update -v $template_latest_tag
    out=$(footing update -v $template_latest_tag 2>&1)
    echo "$out"
    if [[ $out == *"No updates have"* ]]; then
        exit 0
    fi
    if [[ $out == *"ExistingBranchError"* ]]; then
        echo -e $RED"ERROR: Temporary branch ('_footing_update' or '_footing_update_temp') exits."$EOC
        echo To perform 'footing update' from scratch, you can type \'git branch -D \<tmp-branch\>\'
        echo to clean up the temporary branch first.
        exit 1
    fi
    set -ex
    git status
    git add footing.yaml
    git merge --continue
    success=$?
    set +x

    if [ $success -ne 0 ]; then
        echo "Conflicts require inspection."
        echo "Please resolve the conflicts. And 'git add <resolved-files>', and then 'footing-auto --continue'."
        exit 1
    fi
else
    if [[ "$cur_branch" != "_footing_update" ]]; then
        echo -e $RED"ERROR: The current branch is NOT '_footing_update'."$EOC
        echo "To continue to finish footing update, you must switch into the '_footing_update branch".
        exit 1
    fi
    set -ex
    git -c core.editor=true merge --continue || true
    cur_branch=$(git show-branch -a | awk -F'[]^~[]' '/^\*/ && !/'"$cur_branch"'/ {print $2;exit}')
fi

git checkout $cur_branch
git merge _footing_update --no-edit
git branch -d _footing_update
set +x

echo -e $GREEN"Footing updated successfully."$EOC
