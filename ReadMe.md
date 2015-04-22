Pylearn2 Extra
--------------

This repository contains code that extends the [Pylearn2][1] library,
a Machine Learning library based on [Theano][2].

The structure of the code tries to mimic that of the
[Pylearn2][1] library. Some support files (
LICENSE.txt, MANIFEST.in, setup.py, etc) were reused.

[Pylearn2][1] was also added as a submodule, so you can:

    git clone https://github.com/TNick/pyl2extra.git
    cd pyl2extra
    git submodule update --init

The `setup.py` script will look for `pylearn2` directory and, if present,
will call it just like you would:

    cd pylearn2
    python setup.py develop --user

Please note that I am in no way affiliated with
the folks at [Lisa lab][3] that developed [Pylearn2][1] library.

Highlights
----------

This section gives some hints about the content. It will probably
get out of sync pretty fast, so please check the content of the package.




  [1]: https://github.com/lisa-lab/pylearn2 "Pylearn2 GitHub repository"
  [2]: https://github.com/Theano/Theano "Theano - define, optimize, and evaluate mathematical expressions"
  [3]: http://www.iro.umontreal.ca/~lisa/ "Laboratoire d’Informatique des Systèmes Adaptatifs"
