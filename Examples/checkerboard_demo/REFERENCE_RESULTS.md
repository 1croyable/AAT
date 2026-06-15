# Checkerboard Demo Reference Results

These are reference runs for the checkerboard demo.  
They are not strict benchmark results because the example scripts do not fix random seeds.

## train_2D.py

```text
epoch=1 loss=0.8058 train_acc=0.5149 val_acc=0.4780 best_val=0.4780@1
epoch=50 loss=0.7478 train_acc=0.5149 val_acc=0.4780 best_val=0.4780@1
epoch=100 loss=0.6814 train_acc=0.5752 val_acc=0.5327 best_val=0.5366@90
epoch=150 loss=0.5618 train_acc=0.6816 val_acc=0.6548 best_val=0.6548@150
epoch=200 loss=0.4502 train_acc=0.8269 val_acc=0.8145 best_val=0.8218@194
epoch=250 loss=0.3518 train_acc=0.8896 val_acc=0.8936 best_val=0.8936@250
epoch=300 loss=0.2699 train_acc=0.8989 val_acc=0.8931 best_val=0.9009@298
epoch=350 loss=0.2186 train_acc=0.9612 val_acc=0.9487 best_val=0.9536@344
epoch=400 loss=0.2137 train_acc=0.9524 val_acc=0.9507 best_val=0.9551@379
epoch=450 loss=0.2057 train_acc=0.9602 val_acc=0.9458 best_val=0.9551@379
epoch=500 loss=0.1508 train_acc=0.9722 val_acc=0.9595 best_val=0.9609@494
epoch=550 loss=0.1569 train_acc=0.9678 val_acc=0.9639 best_val=0.9683@521
epoch=600 loss=0.1226 train_acc=0.9714 val_acc=0.9604 best_val=0.9683@521
epoch=650 loss=0.1346 train_acc=0.9651 val_acc=0.9561 best_val=0.9683@521
epoch=700 loss=0.1045 train_acc=0.9768 val_acc=0.9541 best_val=0.9683@521
epoch=750 loss=0.1415 train_acc=0.9614 val_acc=0.9448 best_val=0.9683@521
epoch=800 loss=0.1512 train_acc=0.9592 val_acc=0.9487 best_val=0.9683@521
best_epoch: 521
best_train_acc: 0.9778
best_val_acc: 0.9683
selected_children: [[12, 12], [11, 11], [12, 12], [12, 12], [11, 11], [12, 12], [12, 12], [10, 10]]
total_children: 184
```

## train_3D.py

```text
epoch=1 loss=0.7155 train_acc=0.4976 val_acc=0.4800 best_val=0.4800@1
epoch=50 loss=0.6652 train_acc=0.6270 val_acc=0.6250 best_val=0.6250@50
epoch=100 loss=0.5381 train_acc=0.9673 val_acc=0.9541 best_val=0.9541@100
epoch=150 loss=0.3314 train_acc=0.9668 val_acc=0.9673 best_val=0.9756@124
epoch=200 loss=0.2132 train_acc=0.9529 val_acc=0.9502 best_val=0.9756@124
epoch=250 loss=0.2451 train_acc=0.9294 val_acc=0.9287 best_val=0.9756@124
epoch=300 loss=0.1649 train_acc=0.9536 val_acc=0.9507 best_val=0.9756@124
epoch=350 loss=0.1791 train_acc=0.9360 val_acc=0.9229 best_val=0.9756@124
epoch=400 loss=0.1154 train_acc=0.9612 val_acc=0.9512 best_val=0.9756@124
epoch=450 loss=0.1300 train_acc=0.9514 val_acc=0.9434 best_val=0.9756@124
epoch=500 loss=0.1127 train_acc=0.9636 val_acc=0.9644 best_val=0.9756@124
epoch=550 loss=0.1077 train_acc=0.9670 val_acc=0.9673 best_val=0.9756@124
epoch=600 loss=0.0983 train_acc=0.9685 val_acc=0.9702 best_val=0.9756@124
epoch=650 loss=0.0831 train_acc=0.9749 val_acc=0.9707 best_val=0.9761@647
epoch=700 loss=0.0604 train_acc=0.9829 val_acc=0.9805 best_val=0.9829@695
epoch=750 loss=0.0485 train_acc=0.9875 val_acc=0.9834 best_val=0.9878@728
epoch=800 loss=0.1221 train_acc=0.9556 val_acc=0.9497 best_val=0.9878@728
best_epoch: 728
best_train_acc: 0.9890
best_val_acc: 0.9878
selected_children: [[12, 12], [12, 12], [12, 12], [12, 12], [12, 12], [12, 12], [12, 12], [12, 12]]
total_children: 192
```