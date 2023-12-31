Contributing Guide
==================

This project was created using footing.
For more information about footing, go to the
`footing docs <https://github.com/Opus10/footing>`_.

Setup
~~~~~

Set up your development environment with::

    git clone git@gitlab.com:eric.clyang521/atm.git
    cd atm
    PYTHONPATH=. python dev/setup.py

The ``dev/setup.py`` script uses `conda <https://conda.io>`__ to set up a local
development environment. If you don't have conda installed, try installing
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ first.


Testing and Validation
~~~~~~~~~~~~~~~~~~~~~~

Run the tests with::

    python dev/test.py

Validate the code with::

    python dev/lint.py

Run automated code formatting with::

    python dev/lint.py --fix

.. _Structured Notes:

Structured Notes in Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This project uses `detail <https://www.github.com/Opus10/detail>`__ to attach
structured notes when contributing changes. Structured notes are used to
create the ``CHANGELOG.md`` file and properly bump the version of the package
when it is deployed.

When contributing a change, type ``detail`` to add a structured note and commit
it when finished.
``detail`` will prompt you for all relevant information about your contribution
based on the schema in the ``.detail/schema.yaml`` file.

If you're making a change that breaks a user-facing API, choose
``api-break`` as the type of contribution. If you're adding functionality,
choose ``feature``. Choose ``bug`` for bug fixes.

For trivial changes, such as minor fixes, documentation changes, or things that
have little to no impact on a user, choose ``trivial``.

Remember, what you provide in these notes will be surfaced in release notes to
end users. Keep your audience in mind!

Documentation
~~~~~~~~~~~~~

`Sphinx <http://www.sphinx-doc.org/>`_ documentation can be built with::

    python dev/docs.py

The static HTML files are stored in the ``docs/_build/html`` directory.
A shortcut for opening them is::

    python dev/docs.py --open

For a primer on how to write documentation using Sphinx and reStructuredText,
see `this page <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__.
A more human-friendly tutorial is also located
`here <https://sphinx-tutorial.readthedocs.io/step-1/>`__.

Releases and Versioning
~~~~~~~~~~~~~~~~~~~~~~~

Anything that is merged into the ``main`` branch will be automatically deployed
to private conda and pip indices. The conda registry is hosted at Cloudsmith:
https://token:TOKEN@conda.cloudsmith.io/drailab/reg/. See
`the online help here <https://help.cloudsmith.io/docs/conda-repository>`__ for
configuration.

The ``CHANGELOG.md`` is automatically generated from all of the ``detail`` notes
added during the release. It should not be edited manually. Any edits will be
overwritten during deployment. Please be mindful of what notes are written in ``detail``
notes since these will be user facing!

This project uses `Semantic Versioning <http://semver.org>`__ by analyzing
the ``type`` key of notes generated by ``detail``. See the
*Structured Notes in Contributions* section above for more information on
how to choose the type when creating a note.

In order to create an alpha release of a package, tag it with the following structure::

    {MAJOR}.{MINOR}.{PATCH}a{BUILD}

If you tag a non-main branch with this structure (e.g. ``0.0.2a2``) and
``git push --tags``, the tagged code will be released to the package indices and
documentation website using the version you tagged.

All tagging and deployments to the ``main`` branch are handled by Gitlab's CI/CD. They
cannot be performed manually.
