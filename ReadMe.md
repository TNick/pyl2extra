Pylearn2 Extra
==============

[![Join the chat at https://gitter.im/TNick/pyl2extra](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/TNick/pyl2extra?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This repository contains code that extends the [Pylearn2][1] library,
a Machine Learning library based on [Theano][2].

The structure of the code tries to mimic that of the
[Pylearn2][1] library. Some support files (
`LICENSE.txt`, `MANIFEST.in`, `setup.py`, etc) were reused.

[Pylearn2][1] was also added as a submodule, so you can:

    git clone https://github.com/TNick/pyl2extra.git
    cd pyl2extra
    git submodule update --init

The `setup.py` script will look for `pylearn2` directory and, if present,
will call it just like you would:

    cd pylearn2
    python setup.py develop --user

Sphynx generated documentation is available at [tnick.github.io/pyl2extra][8].

Please note that I am in no way affiliated with
the folks at [Lisa lab][3] that developed [Pylearn2][1] library.

Highlights
----------

This section gives some hints about the content. It will probably
get out of sync pretty fast, so please check the content of the package.

- [yaml_parser][9]: some goodies were added to allow extracting
values from classes and ranges of values that generate a new value
at each instantiation.
- [show_weights][10]: a GUI module that allows you to explore the weights of
a model by using [pyqtgraph][5]

Upcoming Features
-----------------

A visual debugger is in the making that allows
you to visualize the parameters
of your model, images in the dataset, run it one
epoch at a time, track the progress.
The front-end GUI interface may run on a different machine than
the one doing actual training.

The debugger uses [PyQt4][4] for GUI, [PyQtGraph][5] for nice graphs
and [ZMQ][6] for networking.


  [1]: https://github.com/lisa-lab/pylearn2 "Pylearn2 GitHub repository"
  [2]: https://github.com/Theano/Theano "Theano - define, optimize, and evaluate mathematical expressions"
  [3]: http://www.iro.umontreal.ca/~lisa/ "Laboratoire d’Informatique des Systèmes Adaptatifs"
  [4]: http://pyqt.sourceforge.net/ "PyQt4"
  [5]: http://www.pyqtgraph.org/ "PyQtGraph - Scientific Graphics and GUI Library for Python"
  [6]: https://github.com/zeromq/pyzmq "GitHub repository for pyzmq"
  [7]: http://sphinx-doc.org/ "Sphinx - Python Documentation Generator"
  [8]: http://tnick.github.io/pyl2extra/ "Pyl2Extra Documentation"
  [9]: https://github.com/TNick/pyl2extra/blob/master/pyl2extra/config/yaml_parse.py
  [10]: https://github.com/TNick/pyl2extra/blob/master/pyl2extra/scripts/show_weights.py
