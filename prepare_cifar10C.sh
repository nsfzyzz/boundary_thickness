git clone https://github.com/google-research/augmix.git
cd augmix
mkdir -p ./data/cifar
curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
curl -O https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
tar -xvf CIFAR-100-C.tar -C data/cifar/
tar -xvf CIFAR-10-C.tar -C data/cifar/