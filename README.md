neuZ - Neural Network library
=================================================================
Copyright (C) Tomomichi Sugihara (Zhidao) since 2020

-----------------------------------------------------------------
## [What is this?]

neuZ is a neural network library written in C.

ZEDA and ZM is required to be installed.

-----------------------------------------------------------------
## [Installation / Uninstallation]

### install

Install ZEDA and ZM in advance.

Move to a directly under which you want to install neuZ, and run:

   ```
   % git clone https://github.com/zhidao/neuz.git
   % cd neuz
   ```

Edit **PREFIX** in *config* file if necessary in order to specify
a directory where the header files, the library and some utilities
are installed. (default: ~/usr)

   - header files: $PREFIX/include/neuz
   - library file: $PREFIX/lib
   - utilities: $PREFIX/bin

Then, make and install.

   ```
   % make && make install
   ```

### uninstall

Do:

   ```
   % make uninstall
   ```

which removes $PREFIX/lib/libneuz.so and $PREFIX/include/neuz.

-----------------------------------------------------------------
## [How to use]

When you want to compile your code *test.c*, for example, the following line will work.

   ```
   % gcc `neuz-config -L` `neuz-config -I` test.c `neuz-config -l`
   ```

-----------------------------------------------------------------
## [Contact]

zhidao@ieee.org
