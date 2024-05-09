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

data[0]['input_data']['de8acb0b-c063-46dd-a2e2-f93230418fcf_page_0_region_1']['image'].save('tmp.png')

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224), ),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)],)
tokenizer = BERTTokenizer()
dataloader = DataLoader(data, collate_fn=Collator(transforms, tokenizer).collate_fn, batch_size=2)

text_model = TransformerTextEncoder(len(tokenizer), 18, 1,1 ).cpu()
visual_model = CLIPVisionEncoder().cpu()
graph_model = GraphConv(18, 768, 10, 20, 3, 10)

for batch in dataloader:

    with torch.no_grad():
        text_features = text_model(batch['textual_content'])
        image_features = visual_model(batch['images'])

        node_features = graph_model(image_features, text_features, batch['input_indices'], batch['gt_indices'], batch['edges'])
        print([x.shape for x in node_features.values()])
    exit()