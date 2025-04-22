## Next steps

- Save highest accuracy child model
- Clean up logging
- Code clean up


## Experiments

```aiignore

# Benchmark Model used for all exp

3 layer CNN - 3 layer MLP model (custom base architecture)
```

### Global

| Model Architecture | 10 Epochs | 20 Epochs | 30 Epochs |
|--------------------|----|----|-----------|
| *VGG11*            | - | - | -         |
| *LeNet*            | - | - | -         |
| *AlexNet*          | - | - | -         |
| *Custom Model*     | - | - | -         |

Train each predefined model for 

### Exp 1
### Exp 2
### Exp 3

Can a controller use dataset metadata information to converge faster or find better models

> Model iters = 300
> 
> Train epochs = 30

#### Steps

1. Train controller on CIFAR-10. **Use Meta param**
2. Save. Plot results
   1. validation accuracy
   2. validation loss
   3. policy gradient
3. Train new controller for CIFAR-10. **Don't use Meta para**
4. Save. Plot results
   1. validation accuracy
   2. validation loss
   3. policy gradient


### Exp 4

Can a controller trained on different dataset converge faster towards a feasible model on a similar dataset

#### Steps
> Model iters = 300
> 
> Train epochs = 30



1. Train controller on CIFAR-10
2. Save. Plot results
   1. validation accuracy
   2. validation loss
   3. policy gradient
3. Run controller for CIFAR-100
4. Save. Plot results
   1. validation accuracy
   2. validation loss
   3. policy gradient
5. Run fresh controller on CIFAR-100
6. Save. Plot results
   1. validation accuracy
   2. validation loss
   3. policy gradient