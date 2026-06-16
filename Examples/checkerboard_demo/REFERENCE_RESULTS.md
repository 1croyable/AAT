# Checkerboard Demo Reference Results

These are reference runs for the checkerboard demo.  
They are not strict benchmark results because the example scripts do not fix random seeds.

## train_2D.py
Across the three 2D checkerboard runs, the average best validation accuracy is 95.90%.

```text
epoch=1 loss=0.7691 train_acc=0.4792 val_acc=0.4771 best_val=0.4771@1
epoch=50 loss=0.7174 train_acc=0.4976 val_acc=0.4912 best_val=0.4912@50
epoch=100 loss=0.6752 train_acc=0.5742 val_acc=0.5684 best_val=0.5684@100
epoch=150 loss=0.6295 train_acc=0.6311 val_acc=0.6216 best_val=0.6260@134
epoch=200 loss=0.5675 train_acc=0.6943 val_acc=0.6797 best_val=0.6821@197
epoch=250 loss=0.4700 train_acc=0.7761 val_acc=0.7583 best_val=0.7583@250
epoch=300 loss=0.4037 train_acc=0.8831 val_acc=0.8711 best_val=0.8711@300
epoch=350 loss=0.3302 train_acc=0.8889 val_acc=0.8711 best_val=0.8755@304
epoch=400 loss=0.2933 train_acc=0.8962 val_acc=0.8755 best_val=0.8770@392
epoch=450 loss=0.2472 train_acc=0.9507 val_acc=0.9326 best_val=0.9326@446
epoch=500 loss=0.2061 train_acc=0.9668 val_acc=0.9424 best_val=0.9463@482
epoch=550 loss=0.1858 train_acc=0.9680 val_acc=0.9473 best_val=0.9541@546
epoch=600 loss=0.1520 train_acc=0.9761 val_acc=0.9575 best_val=0.9575@599
epoch=650 loss=0.1344 train_acc=0.9775 val_acc=0.9609 best_val=0.9629@649
epoch=700 loss=0.1188 train_acc=0.9805 val_acc=0.9624 best_val=0.9629@649
epoch=750 loss=0.1066 train_acc=0.9807 val_acc=0.9619 best_val=0.9658@713
epoch=800 loss=0.0995 train_acc=0.9829 val_acc=0.9604 best_val=0.9658@713
saved best checkpoint: C:\Projets\AATField\Examples\checkerboard_demo\checkerboard_2d.pt
best_epoch: 713
best_train_acc: 0.9829
best_val_acc: 0.9658
selected_children: [[8, 8], [8, 8], [8, 8], [8, 8]]
total_children: 64
```

```text
epoch=1 loss=0.7244 train_acc=0.5095 val_acc=0.5034 best_val=0.5034@1
epoch=50 loss=0.6948 train_acc=0.4929 val_acc=0.4873 best_val=0.5039@2
epoch=100 loss=0.6709 train_acc=0.5425 val_acc=0.5327 best_val=0.5327@100
epoch=150 loss=0.6364 train_acc=0.6270 val_acc=0.6030 best_val=0.6030@150
epoch=200 loss=0.5566 train_acc=0.7070 val_acc=0.6787 best_val=0.6787@200
epoch=250 loss=0.4871 train_acc=0.7598 val_acc=0.7417 best_val=0.7422@249
epoch=300 loss=0.4305 train_acc=0.7959 val_acc=0.7764 best_val=0.7769@298
epoch=350 loss=0.3737 train_acc=0.8447 val_acc=0.8223 best_val=0.8223@350
epoch=400 loss=0.3170 train_acc=0.9111 val_acc=0.8960 best_val=0.8960@400
epoch=450 loss=0.2595 train_acc=0.9275 val_acc=0.9004 best_val=0.9053@442
epoch=500 loss=0.2162 train_acc=0.9475 val_acc=0.9292 best_val=0.9292@499
epoch=550 loss=0.1785 train_acc=0.9609 val_acc=0.9360 best_val=0.9380@530
epoch=600 loss=0.1615 train_acc=0.9607 val_acc=0.9380 best_val=0.9424@594
epoch=650 loss=0.1393 train_acc=0.9646 val_acc=0.9390 best_val=0.9438@606
epoch=700 loss=0.1300 train_acc=0.9666 val_acc=0.9414 best_val=0.9468@673
epoch=750 loss=0.1377 train_acc=0.9592 val_acc=0.9404 best_val=0.9487@743
epoch=800 loss=0.1436 train_acc=0.9558 val_acc=0.9297 best_val=0.9487@743
saved best checkpoint: C:\Projets\AATField\Examples\checkerboard_demo\checkerboard_2d.pt
best_epoch: 743
best_train_acc: 0.9656
best_val_acc: 0.9487
selected_children: [[8, 8], [8, 8], [8, 8], [8, 8]]
total_children: 64
```

```text
epoch=1 loss=0.7790 train_acc=0.4971 val_acc=0.4644 best_val=0.4644@1
epoch=50 loss=0.7327 train_acc=0.4971 val_acc=0.4644 best_val=0.4644@1
epoch=100 loss=0.6963 train_acc=0.4749 val_acc=0.4531 best_val=0.4644@1
epoch=150 loss=0.6289 train_acc=0.6328 val_acc=0.6118 best_val=0.6123@149
epoch=200 loss=0.5177 train_acc=0.7788 val_acc=0.7607 best_val=0.7607@200
epoch=250 loss=0.4057 train_acc=0.8694 val_acc=0.8613 best_val=0.8613@250
epoch=300 loss=0.3316 train_acc=0.9446 val_acc=0.9292 best_val=0.9292@297
epoch=350 loss=0.2707 train_acc=0.9563 val_acc=0.9414 best_val=0.9438@344
epoch=400 loss=0.2220 train_acc=0.9656 val_acc=0.9487 best_val=0.9526@379
epoch=450 loss=0.2001 train_acc=0.9651 val_acc=0.9512 best_val=0.9526@379
epoch=500 loss=0.1680 train_acc=0.9714 val_acc=0.9541 best_val=0.9561@472
epoch=550 loss=0.1472 train_acc=0.9714 val_acc=0.9541 best_val=0.9580@537
epoch=600 loss=0.1252 train_acc=0.9719 val_acc=0.9512 best_val=0.9585@554
epoch=650 loss=0.1054 train_acc=0.9766 val_acc=0.9570 best_val=0.9585@554
epoch=700 loss=0.0874 train_acc=0.9783 val_acc=0.9580 best_val=0.9609@688
epoch=750 loss=0.0787 train_acc=0.9812 val_acc=0.9585 best_val=0.9624@728
epoch=800 loss=0.0758 train_acc=0.9844 val_acc=0.9580 best_val=0.9624@728
saved best checkpoint: C:\Projets\AATField\Examples\checkerboard_demo\checkerboard_2d.pt
best_epoch: 728
best_train_acc: 0.9834
best_val_acc: 0.9624
selected_children: [[8, 8], [8, 8], [8, 8], [8, 8]]
total_children: 64
```

---


## train_3D.py
Across the three 3D checkerboard runs, the average best validation accuracy is 97.85%.

```text
epoch=1 loss=0.7485 train_acc=0.4963 val_acc=0.5132 best_val=0.5132@1
epoch=50 loss=0.6992 train_acc=0.4963 val_acc=0.5132 best_val=0.5132@1
epoch=100 loss=0.6227 train_acc=0.6758 val_acc=0.6699 best_val=0.6699@100
epoch=150 loss=0.4973 train_acc=0.9407 val_acc=0.9390 best_val=0.9390@150
epoch=200 loss=0.3056 train_acc=0.9734 val_acc=0.9663 best_val=0.9688@199
epoch=250 loss=0.1907 train_acc=0.9702 val_acc=0.9556 best_val=0.9688@199
epoch=300 loss=0.1588 train_acc=0.9675 val_acc=0.9551 best_val=0.9688@199
epoch=350 loss=0.1860 train_acc=0.9500 val_acc=0.9326 best_val=0.9688@199
epoch=400 loss=0.1073 train_acc=0.9795 val_acc=0.9639 best_val=0.9688@199
epoch=450 loss=0.0960 train_acc=0.9788 val_acc=0.9722 best_val=0.9727@436
epoch=500 loss=0.0973 train_acc=0.9722 val_acc=0.9639 best_val=0.9741@473
epoch=550 loss=0.0616 train_acc=0.9839 val_acc=0.9663 best_val=0.9741@473
epoch=600 loss=0.0648 train_acc=0.9822 val_acc=0.9663 best_val=0.9751@585
epoch=650 loss=0.1145 train_acc=0.9644 val_acc=0.9556 best_val=0.9775@619
epoch=700 loss=0.0479 train_acc=0.9873 val_acc=0.9741 best_val=0.9775@619
epoch=750 loss=0.0438 train_acc=0.9888 val_acc=0.9746 best_val=0.9795@729
epoch=800 loss=0.0397 train_acc=0.9878 val_acc=0.9722 best_val=0.9795@729
saved best checkpoint: C:\Projets\AATField\Examples\checkerboard_demo\checkerboard_3d.pt
best_epoch: 729
best_train_acc: 0.9912
best_val_acc: 0.9795
selected_children: [[8, 8], [8, 8], [8, 8], [8, 8]]
total_children: 64
```

```text
epoch=1 loss=0.7420 train_acc=0.5217 val_acc=0.5210 best_val=0.5210@1
epoch=50 loss=0.6982 train_acc=0.5022 val_acc=0.5015 best_val=0.5210@1
epoch=100 loss=0.6357 train_acc=0.6907 val_acc=0.6738 best_val=0.6738@100
epoch=150 loss=0.5176 train_acc=0.8855 val_acc=0.8677 best_val=0.8677@150
epoch=200 loss=0.3849 train_acc=0.9631 val_acc=0.9463 best_val=0.9492@196
epoch=250 loss=0.2771 train_acc=0.9602 val_acc=0.9419 best_val=0.9556@245
epoch=300 loss=0.2202 train_acc=0.9690 val_acc=0.9585 best_val=0.9585@300
epoch=350 loss=0.2009 train_acc=0.9600 val_acc=0.9482 best_val=0.9595@301
epoch=400 loss=0.1463 train_acc=0.9795 val_acc=0.9600 best_val=0.9604@395
epoch=450 loss=0.1487 train_acc=0.9678 val_acc=0.9531 best_val=0.9692@430
epoch=500 loss=0.1226 train_acc=0.9797 val_acc=0.9644 best_val=0.9692@430
epoch=550 loss=0.1177 train_acc=0.9763 val_acc=0.9619 best_val=0.9692@430
epoch=600 loss=0.0925 train_acc=0.9822 val_acc=0.9639 best_val=0.9736@586
epoch=650 loss=0.0810 train_acc=0.9834 val_acc=0.9751 best_val=0.9751@650
epoch=700 loss=0.1138 train_acc=0.9744 val_acc=0.9604 best_val=0.9751@650
epoch=750 loss=0.1006 train_acc=0.9771 val_acc=0.9614 best_val=0.9751@650
epoch=800 loss=0.0932 train_acc=0.9792 val_acc=0.9692 best_val=0.9751@650
saved best checkpoint: C:\Projets\AATField\Examples\checkerboard_demo\checkerboard_3d.pt
best_epoch: 650
best_train_acc: 0.9834
best_val_acc: 0.9751
selected_children: [[8, 8], [8, 8], [8, 8], [8, 8]]
total_children: 64
```

```text
C:\Projets\AATField\.venv\Lib\site-packages\torch\_subclasses\functional_tensor.py:362: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
epoch=1 loss=0.6954 train_acc=0.4873 val_acc=0.4883 best_val=0.4883@1
epoch=50 loss=0.6860 train_acc=0.5664 val_acc=0.5664 best_val=0.5664@50
epoch=100 loss=0.6309 train_acc=0.6880 val_acc=0.6792 best_val=0.6792@100
epoch=150 loss=0.4513 train_acc=0.9033 val_acc=0.8945 best_val=0.8965@147
epoch=200 loss=0.2864 train_acc=0.9612 val_acc=0.9556 best_val=0.9629@195
epoch=250 loss=0.2088 train_acc=0.9617 val_acc=0.9556 best_val=0.9629@195
epoch=300 loss=0.2113 train_acc=0.9480 val_acc=0.9424 best_val=0.9629@195
epoch=350 loss=0.1612 train_acc=0.9639 val_acc=0.9473 best_val=0.9629@195
epoch=400 loss=0.1485 train_acc=0.9629 val_acc=0.9468 best_val=0.9629@195
epoch=450 loss=0.1130 train_acc=0.9758 val_acc=0.9580 best_val=0.9648@421
epoch=500 loss=0.1572 train_acc=0.9551 val_acc=0.9380 best_val=0.9648@421
epoch=550 loss=0.1158 train_acc=0.9719 val_acc=0.9604 best_val=0.9648@421
epoch=600 loss=0.0979 train_acc=0.9783 val_acc=0.9663 best_val=0.9741@579
epoch=650 loss=0.0810 train_acc=0.9822 val_acc=0.9712 best_val=0.9741@579
epoch=700 loss=0.0626 train_acc=0.9883 val_acc=0.9756 best_val=0.9780@669
epoch=750 loss=0.0686 train_acc=0.9854 val_acc=0.9766 best_val=0.9780@669
epoch=800 loss=0.0995 train_acc=0.9707 val_acc=0.9595 best_val=0.9810@762
saved best checkpoint: C:\Projets\AATField\Examples\checkerboard_demo\checkerboard_3d.pt
best_epoch: 762
best_train_acc: 0.9861
best_val_acc: 0.9810
selected_children: [[8, 8], [8, 8], [8, 8], [8, 8]]
total_children: 64
```