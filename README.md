**Breakthrough AES Performance on GPUs**

These are CUDA optimizations of T-table based implementation of AES which contain zero bank conflicts. 

We achieved 

**315.2** Gbps AES-128 encryption on a **GTX 970**<br>
**878.6** Gbps AES-128 encryption on an **RTX 2070 Super**

These results are published in https://ieeexplore.ieee.org/document/9422754

In science, reproducibility of experiments is crucial but almost none of the GPU optimizations of AES is publicly availble. This is why we publish our codes here. 

Moreover, comparing different optimization results on different GPUs is almost impossible. When you have adifferent kind of optimization and want to compare it with our optimizations, please use these codes on the same GPU you used for your codes.


**Cihangir Tezcan**, PhD<br>
_Director of Cyber Security Center_<br>
_Head of Department of Cyber Security, Informatics Institute_<br>
_Middle East Technical University_<br>
_Ankara, Turkey_
