# elements of style

Important things:
1. Python 3
2. Make a folder at the same level as this called data. We have added it to the gitignore.
3. Datasets are stored at data/<dataset-name>

## Data loading

We will make use of `torchvision.datasets`, specifically the `ImageFolder` implementation of the `Dataset` class. Assuming image data is arranged as follows:
```
root/<label1>/image00.png
root/<label1>/image01.png
...
root/<labelN>/imageXX.png
```
we can then load our dataset like so:
```python
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([transform1, transform2, ...]) # apply some transformations
bam_dataset = datasets.ImageFolder(root="data/bam/", transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(bam_dataset, batch_size=batch_sz, shuffle=True)
```


Helpful commands:
python3 mnist_fc_vae_experiments.py -res=results/MNIST/train_z_sz_10/ -save=trained_models/mnist_fc_vae_z_sz_10.model -z_sz=10 -epochs=50
