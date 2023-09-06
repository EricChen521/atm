#!/bin/bash

set -ex
echo ">>>>>>>>>> lint..."
python dev/run.py lint.py -b $DETAIL_AGAINST
echo ">>>>>>>>>> test..."
python dev/run.py test.py
if [[ $CI_COMMIT_BRANCH == $MAIN_BRANCH ]]; then
    LATEST_TAG=$(git ls-remote -tq --refs --sort=-v:refname | head -1 | cut -f 3 -d /)
    echo ">>>>>>>>>> tag..."
    python dev/run.py deploy.py tag
    diff_docs=$(git diff --name-only $LATEST_TAG.. docs)
    if [[ $diff_docs != "" ]]; then
        echo ">>>>>>>>>> pages..."
        python dev/run.py deploy.py docs
    fi
    echo ">>>>>>>>>> pip-package..."
    python dev/run.py deploy.py pip-package
    echo ">>>>>>>>>> conda-package..."
    python dev/run.py deploy.py conda-package \"$BUILD_PYVERS\"
fi
echo "Done"
set +x
