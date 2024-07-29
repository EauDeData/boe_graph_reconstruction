from src.data.datasets import BOEDataset
from torch.utils.data import DataLoader
from src.data.collator import Collator
import torchvision
from src.data.defaults import IMAGENET_STDS, IMAGENET_MEANS
from src.tokenizer.text import BERTTokenizer
from src.models.text import TransformerTextEncoder
from src.models.vision import CLIPVisionEncoder
from src.models.graphs import GraphConv
import torch


data = BOEDataset('test.txt')
print(len(data))

#print(data[0])

# data[0]['input_data']['de8acb0b-c063-46dd-a2e2-f93230418fcf_page_0_region_1']['image'].save('tmp.png')

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224), ),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)],)
tokenizer = BERTTokenizer()
dataloader = DataLoader(data, collate_fn=Collator(transforms, tokenizer).collate_fn, batch_size=2)

for batch in dataloader:
    print(batch)
    exit()