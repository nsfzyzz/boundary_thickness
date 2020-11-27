mkdir -p ../data/ood/cifar
curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
curl -O https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
tar -xvf CIFAR-100-C.tar -C ../data/ood/cifar/
tar -xvf CIFAR-10-C.tar -C ../data/ood/cifar/
rm CIFAR-100-C.tar
rm CIFAR-10-C.tar