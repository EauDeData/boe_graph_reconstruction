import torch
from torch.utils.data import DataLoader
from src.data.datasets import BOEDataset
from src.tokenizer.text import BERTTokenizer
from src.data.collator import Collator
from main import load_models
from src.models.utils import JointModel
import torch.nn as nn
import json
import open_clip
from tqdm import tqdm
import random
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
del model
del _

tokenizer = BERTTokenizer(context_length=77)
collator = Collator(preprocess, tokenizer)

distractor_set = BOEDataset('distractorsV2.txt', random_walk_leng=0)
distractor_dataloader = DataLoader(distractor_set,
                       collate_fn=collator.collate_fn,
                       num_workers=36,
                       batch_size=1,
                       shuffle=False)
def cosine_similarity_matrix(x1, x2, dim: int = 1, eps: float = 1e-8):
    '''
    When using cosine similarity the constant value must be positive
    '''
    #Cosine sim:
    xn1, xn2 = torch.norm(x1, dim=dim), torch.norm(x2, dim=dim)
    x1 = x1 / torch.clamp(xn1, min=eps).unsqueeze(dim)
    x2 = x2 / torch.clamp(xn2, min=eps).unsqueeze(dim)
    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(1)

    sim = torch.tensordot(x1, x2, dims=([2], [2])).squeeze()

    sim = (sim + 1)/2 #range: [-1, 1] -> [0, 2] -> [0, 1]

    return sim
class CosineSimilarityMatrix(nn.Module):
    name = 'cosine_matrix'
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarityMatrix, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return cosine_similarity_matrix(x1, x2, self.dim, self.eps)

cosine_matrix = CosineSimilarityMatrix()

class Args:
    def __init__(self):
        return

json_path = '/data2/users/amolina/leviatan/structured_approaches/gnn_graph_only_sample_one_neigh.json'
with open(json_path, 'r') as file:
    data = json.load(file)

# Initialize the Args class
args = Args()

# Set attributes using setattr
for key, value in data.items():
    setattr(args, key, value)

text_model, visual_model, graph_model, queries_model = load_models(args, collator)
joint_model = JointModel(visual_model, text_model, graph_model, queries_model)

model_path = '/data2/users/amolina/leviatan/structured_approaches/gnn_graph_only_sample_one_neigh.pth'
joint_model.load_state_dict(torch.load(model_path, map_location='cpu'))
joint_model.eval().to('cuda')
print(joint_model)

distractor_embeddings = []
distractor_queries = []

with torch.no_grad():
    for batch in tqdm(distractor_dataloader, total=len(distractor_dataloader), desc='Extracting distractor set...'):

        try:
            visual, query = tuple(joint_model(batch).values())
        except: continue
        distractor_embeddings.append(visual.to('cpu').squeeze())
        distractor_queries.append(query.to('cpu').squeeze())

accepted_num = len(distractor_queries)
accepted = random.sample(list(range(len(distractor_queries))), accepted_num)


distractor_embeddings = torch.stack(distractor_embeddings).to('cuda')[accepted]
distractor_queries = torch.stack(distractor_queries).to('cuda')[accepted]

print("Final distractor size:", distractor_embeddings.size(0))

test_set = BOEDataset('test.txt')
test_dataloader = DataLoader(test_set,
                       collate_fn=collator.collate_fn,
                       num_workers=36,
                       batch_size=1,
                       shuffle=False)


acc_1 = 0
acc_5 = 0
acc_10 = 0

# Todo: This is temporal for the first tries, but please check where the error comes from
correct_tries = 0

with torch.no_grad():
    for batch in tqdm(test_dataloader, total=len(test_dataloader), desc='Comparing with test set...'):

        try:
            visual, query = tuple(joint_model(batch).values())
        except: continue
        correct_tries +=1
        temporal_image_batch = torch.cat((visual, distractor_embeddings), dim=0)
        probs = 1 - cosine_matrix(query, temporal_image_batch)

        poss = probs.argsort().cpu().numpy().tolist()
        position = poss.index(0)

        if position == 0:
            acc_1 += 1
            acc_10 += 1
            acc_5 += 1
        elif position < 5:
            acc_10 += 1
            acc_5 += 1

        elif position < 10:
            acc_10 += 1

print("_______________________________\n test finished with the following metrics:")
print(f"Acc@1: {acc_1/correct_tries}")
print(f"Acc@5: {acc_5/correct_tries}")
print(f"Acc@10: {acc_10/correct_tries}")