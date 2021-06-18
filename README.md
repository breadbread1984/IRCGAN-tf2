# IRC-GAN

## create dataset

create moving single digit dataset with command

```shell
python3 dataset/mnist_caption_single.py
```

after executing successfully, a file named mnist_single_git.h5 is generated.

create moving double digits dataset with command

```shell
python3 dataset/mnist_caption_two_digit.py
```

after executing successfully, a file named mnist_two_gif.h5 is generated.

the dataset creation code is borrowed from [Sync-Draw](https://github.com/syncdraw/Sync-DRAW/tree/master/dataset) and slightly modified.

## train on moving mnist dataset

train model with command

```shell
python3 train.py (single|double)
```

the parameter is an optional to switch between moving single digit and moving double digits.
