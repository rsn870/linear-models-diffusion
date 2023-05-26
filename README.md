# Linear Transformation Experiments

Code base is a modified version of Heat Equation's code. Details are below the dividing line. Similar to heat equation we assume the below model for the forward process

$$u_{t} = T(t)u_{0} + V(t) + n$$ where $T(t)$ is a deterministic linear transformation that depends only on $t$, $V$ is some function of t and $n$ is noise from a constant distribution the values for which will remain the same as the heat equation values for now.

All code for forward processes is present in model_code/utils.py. Code for identity transformation and two kinds of random matrix transformaions has already been added. Follow the template there to add any additional code.

There is a Function_map dictionary present in model_code/utils.py. The options inside '' are the keys or the config/choice of forward process. For any new forward process create similar to existing template and give a name. In any config file located inside configs folder feel free to change the config.model.forward argument to desired key. If said argument is not present in any config file feel free to add in the config.model section. It has already been added for MNIST with value set to identity. Change as per your choice.

Follow the instructions given for mnist below the dividng line.  

## Updates 

- [x] Identity map code works. Results are obtained very fast.
- [x] Fixed random map i.e T(t) = unique random matrix $\forall t \in [0,T]$ code works. But converegence is painfully slow.
- [ ] Alternate fixed random map paramaterization i.e $T(t) = \Pi_{i}R_{i}$ for random matrices $R_{i}$ fails due to numerical instability. 
- [x] Added constant velocity map. Convergence looks good
- [x] Added constant deceleration map. Converegnce looks good
- [x] Added constant velocity based darkening map. Convergence looks good
- [x] Added constant deceleration map with reversal. Testing values of $a$ to check where model can reconstruct. 
- [ ] Added cooling map. Testing to see if sampling works. Sampling fails so far. Need to check why this happens.
- [x] Reimplementing cooling map. This time latent is set to darkened version of image. Test to check if model can learn cooling or latent is the main issue. Model has learnt therefore latent dynamics needs to be adjusted.

## Some simple conclusions so far 

1. In general deterministic linear transformations that depend only on time do seem to be invertible as our hypothesis suggests 
2. Complexity of learning as of now seems to depend on complexity of $T(t)$ as we observed with the edge cases.
3. An interesting experiment to check later would be to extend to the stochastic case where $T(t)$ remains independent of $u$ but instead follows a distribution. This could work as a dry run for the non-linear case.
4. We look set for the next step of constructing $T(t)$ such that the distribution $\{\lim_{t \to \infty}T(t)u_{0}\}$ holds interesting properties

## Physics inspired constructions 

### Mechanics 

1. **Constant Velocity**  We have $$u_{t} = u_{0} -\frac{at}{K} + n$$ where $K$ is the time horizon. This represents a particle at position $u_{0}$ that will reach $u_{0}-a$ at time $K$ with a constant velocity of $\frac{-a}{K}$ throughout. 
2. **Constant Deceleration** We have $$u_{t} = u_{0} -\frac{at}{K} +\frac{at^{2}}{2K^{2}} + n$$ where $K$ is the time horizon. This represents a particle at position $u_{0}$ that will come to rest at $u_{0}-\frac{a}{2}$ at time $K$ with an initial velocity of $\frac{-a}{K}$ and a constant accelaration of $\frac{a}{K^{2}}$ throughout.
3. **Constant Deceleration with reversal** We have $$u_{t} = u_{0} -\frac{at}{K} +\frac{at^{2}}{K^{2}} + n$$ where $K$ is the time horizon. This represents a particle at position $u_{0}$ that will come to  $u_{0}-\frac{a}{2}$  at time $\frac{K}{2}$, reverse and then return to the initial point at time $K$ with an initial velocity of $\frac{-a}{K}$ and a constant accelaration of $\frac{2a}{K^{2}}$ throughout. This mapping is **not** invertible and is designed as a test to check if invertibility is essential.
4. **Constant velocity as varying brightness** Consider the equation $$u_{t} = u_{0} -\frac{bt}{K}u_{0} + n$$ This represents a case where particle reaches $(1-b)u_{0}$ at time $K$ with a constant velocity $-\frac{b}{K}u_{0}$. Practically this is equivalent to darkening the image! 
5. **Velocity as varying brightness with constant deceleration** Consider the equation $$u_{t} = u_{0} -\frac{bt}{K}u_{0} + \frac{bt^{2}}{2K^{2}}u_{0} + n$$ This represents a case where particle comes to rest at $\frac{(1-b)}{2}u_{0}$ at time $K$ with an initial velocity $-\frac{b}{K}u_{0}$ and constany acceleration $\frac{b}{K^{2}}u_{0}$.

In the above interpretations for interpretability in terms of images and numerical stability we need to chose $a$ and $b$ carefully.The displacement parameter can be varied in practice in only a certain range because of this.

### Thermodyanmics 

1. **Newton's Law of Cooling** : We have that the temperature of a system has the following behavior : $$T_{sys}(t) = T_{env} + (T_{sys}(0)-T_{env})e^{\frac{-t}{\tau}}$$ where $\tau$ is a parameter. The key trick here lies in constructing $T_{env}$ which is the limit at $\infty$. We use the following paramterization instead where $K$ is once again the time horizon. $$u_{t} = u_{latent} + (u_{0}-u_{latent})e^{\frac{-t}{K-t}} + n$$ which ensures that we are arbitrarily close to a $u_{latent}$ at the horizon which belongs to a nice distribution $p(.)$. Given that in our paramaterization as of now, dimensions are invariant, we use the Diffusion Model Forward Process as a proxy instead with a fixed random seed for all images to obtain a nice latent space pliable to sampling. This possibly has a neat physical interpretation as well. 
2. **Geometric Brownian Motion** : We set a forward Geometric Brownian Motion SDE with the following discretization : 

$$u_{t} = u_{t-1}e^{r-\frac{\sigma^{2}}{2}}+\sigma n$$

## Results

### Sampled image for identity transformation 

![id sample](https://github.com/rsn870/linear-models-diffusion/blob/main/images/id.jpg?raw=true)

### Sampled image for deceleration with reversal
![dec_rev_samp](https://github.com/rsn870/linear-models-diffusion/blob/main/images/reversal.jpg?raw=true)


### Sampled image for constant velocity darkening
![dark](https://github.com/rsn870/linear-models-diffusion/blob/main/images/darkening.png?raw=true)

## PFGM and alternate formualtions for velocity

PFGM :

* Yilun Xu, Ziming Liu, Max Tegmark, Tommi S. Jaakkola (2022). **Poisson Flow Generative Models**. In *Neurips*. [[arXiv]](https://arxiv.org/pdf/2209.11178.pdf)
This proposes an alternative paramterization :

$$ \frac{du}{dt} = \vec{E} $$

where $\vec{E}$ represents a field in their case the Electric Field where the particles are assumed to have uniform charge of the same kind. Here the model estimates $\vec{E}$ instead. Empirically we have the following formulation during training :

$$\frac{du}{dt} = \frac{u}{\||u\||} $$


------------------------------------------ DIVIDING LINE -----------------------------------

## Generative Modelling With Inverse Heat Dissipation
 
This repository is the official implementation of the methods in the publication:


* Severi Rissanen, Markus Heinonen, and Arno Solin (2023). **Generative Modelling With Inverse Heat Dissipation**. In *International Conference on Learning Representations (ICLR)*. [[arXiv]](https://arxiv.org/abs/2206.13397) [[project page]](https://aaltoml.github.io/generative-inverse-heat-dissipation)


## Arrangement of code

The "`configs`" folder contains the configuration details on different experiments and the "`data`" folder contains the data. MNIST and CIFAR-10 should run as-is with automatic torchvision data loading, but the other experiments require downloading the data to the corresponding `data/` folders. The "`model_code`" contains the U-Net definition and utilities for working with the proposed inverse heat dissipation model. "`scripts`" contains additional code, for i/o, data loading, loss calculation and sampling. "`runs`" is where the results get saved at.

## Used Python packages

The file "requirements.txt" contains the Python packages necessary to run the code, and they can be installed by running

```pip install -r requirements.txt```

If you have issues with installing the `mpi4py` through pip, you can also install it using conda with `conda install -c conda-forge mpi4py`. 

## Training

You can get started by running an MNIST training script with

```python train.py --config configs/mnist/default_mnist_configs.py --workdir runs/mnist/default```

This creates a folder "`runs/mnist/default`", which contains the folder "`checkpoint-meta`", where the newest checkpoint is saved periodically. "`samples`" folder contains samples saved during training. You can change the frequency of checkpointing and sampling with the command line flags "`training.snapshot_freq_for_preemption=?`" and "`config.training.sampling_freq=?`". 

## Sampling
Once you have at least one checkpoint, you can do sampling with "`sample.py`", with different configurations:

### Random samples
Random samples: 

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --batch_size=9
```

### Share the initial state
Samples where the prior state u_K is fixed, but the sampling noise is different:

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --batch_size=9
                 --same_init
```

### Share the noise
Samples where the prior state u_K changes, but the sampling noises are shared (results in similar overall image characteristics, but different average colours if the maximum blur is large enough):

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --batch_size=9
                 --share_noise
 ```

### Interpolation
Produces an interpolation between two random points generated by the model. 

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --interpolate --num_points=20
```

## Evaluation

The script "`evaluation.py`" contains code for evaluating the model with FID-scores and NLL (ELBO) values. For example, if you have a trained cifar-10 model trained with `configs/cifar10/default_cifar10_configs.py` and the result is in the folder `runs/cifar10/default/checkpoints-meta`, you can run the following (checkpoint=0 refers to the last checkpoint, other checkpoints in `runs/cifar10/default/checkpoints` are numbered as 1,2,3,...):

### FID scores
This assumes that you have `clean-fid` installed. 

```bash
python evaluate.py --config configs/cifar10/default_cifar10_configs.py
            --workdir runs/cifar10/default --checkpoint 0
            --dataset_name=cifar10
            --experiment_name=experiment1 --param_name=default --mode=fid
            --delta=0.013 --dataset_name_cleanfid=cifar10
            --dataset_split=train --batch_size=128 --num_gen=50000
```

### NLL values
The result contains a breakdown of the different terms in the NLL.

```bash
python evaluate.py --config configs/cifar10/default_cifar10_configs.py
            --workdir runs/cifar10/default --checkpoint 0
            --dataset_name=cifar10
            --experiment_name=experiment1 --param_name=default --mode=elbo
            --delta=0.013
```

### Result folder
The results will be saved in the folder `runs/cifar10/evaluation_results/experiment1/` in log files, where you can read them out. The idea in general is that `experiment_name` is an upper-level name for a suite of experiments that you might want to have (e.g., FID w.r.t. different delta), and `param_name` is the name of the calculated value within that experiment (e.g., "delta0.013" or "delta0.012"). 

## Citation

If you use the code in this repository for your research, please cite the paper as follows:

```bibtex
@inproceedings{rissanen2023generative,
  title={Generative modelling with inverse heat dissipation},
  author={Severi Rissanen and Markus Heinonen and Arno Solin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

## License

This software is provided under the [MIT license](LICENSE).

