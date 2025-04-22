## Next steps

- Save highest accuracy child model
- Update utils to download CIFAR-100
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

### Exp 1 (R)

Can a controller use dataset metadata information to converge faster or find better models

> Aspect = Changing Controller input
>
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



### Exp 2 (R)

Can we modify reward function to consider both child model accuracy and model complexity

> Aspect = changing Reward FN
> 
> Model iters = 300
> 
> Train epochs = 30



1. Train controller on CIFAR-10 with different reward fn
2. Save. Plot results
   1. validation accuracy
   2. validation loss
   3. policy gradient
3. Compare with pre-existing CIFAR-10 (Use from other exps)



### Exp 3 ()


Can we modify only activation functions


### Exp 4 (V)

Can a controller trained on different dataset converge faster towards a feasible model on a similar dataset

#### Steps
> Aspect = changing dataset
> 
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

Check if pre-trained model reaches best accuracy in a stable fashion before fresh