import networkx as nx
import torch

class Collator:
    def __init__(self, transforms, tokenizer):
        self.tokenizer = tokenizer
        self.transforms = transforms

    def collate_fn(self, batch):
        graphs = nx.compose_all([sample['graph'] for sample in batch])
        input_data_global_dict = {key: value
                                  for d in [sample['input_data'] for sample in batch]
                                  for key, value in d.items()
                                  }
        nodes2idx_lut = {node: idx for idx, node in enumerate(graphs.nodes())}
        # gt_nodes_idxs = []
        input_nodes_idxs = []

        images = []
        text_content = []
        ars = []

        for node, num in nodes2idx_lut.items():
            node_data = graphs.nodes[node]
            if not node_data['is_edge']:
                input_nodes_idxs.append(num)
                image = input_data_global_dict[node]['image']
                text = input_data_global_dict[node]['ocr']

                tensor_image = self.transforms(image)
                tensor_text = self.tokenizer.tokenize([text])[0]

                images.append(tensor_image)
                text_content.append(tensor_text)
                ars.append(torch.tensor((node_data['aspect_ratio'], node_data['width'], node_data['height'])))

            else:
                pass
                # gt_nodes_idxs.append(num)
                # gt.append(torch.tensor([node_data['angle'], node_data['distance']]))

        edges = []
        gt = []
        for edge in graphs.edges():
            node_in, node_out = edge
            edges.extend(
                        [
                            (nodes2idx_lut[node_in], nodes2idx_lut[node_out]),
                         (nodes2idx_lut[node_out], nodes2idx_lut[node_in])]
                         )
            gt.extend([
                torch.tensor(graphs[node_in][node_out]['gt']), torch.tensor(graphs[node_out][node_in]['gt'])
            ])
        gt = torch.stack(gt)
        # gt = ((gt - gt.mean(0)) / (gt.std(0)))

        return {
            'images': torch.stack(images),
            'textual_content': torch.stack(text_content),
            'gt': gt,
            'gt_indices': [], # GT nodes will be inizialized with a learnt token from the embedding
            'input_indices': input_nodes_idxs, # Input indices will be niizialized with a vision and text encoder,
            'edges': torch.tensor(edges),
            'ars': torch.stack(ars)
        }





