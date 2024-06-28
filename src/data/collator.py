import networkx as nx
import torch

class Collator:
    def __init__(self, transforms, tokenizer):
        self.tokenizer = tokenizer
        self.transforms = transforms

    def collate_fn(self, batch):
        graphs = nx.compose_all([sample['graph'] for sample in batch])

        batch_indices = {}
        for batch_idx, sample in enumerate(batch):
            graph = sample['graph']
            for node in graph:
                batch_indices[node] = batch_idx

        input_data_global_dict = {key: value
                                  for d in [sample['input_data'] for sample in batch]
                                  for key, value in d.items()
                                  }
        nodes2idx_lut = {node: idx for idx, node in enumerate(graphs.nodes())}
        # selected_nodes_idxs = [nodes2idx_lut[sample['selected']] for sample in batch]
        input_nodes_idxs = []

        images = []
        text_content = []
        nodes_to_batch = []

        for node, num in nodes2idx_lut.items():

            nodes_to_batch.append(batch_indices[node])
            input_nodes_idxs.append(num)
            image = input_data_global_dict[node]['image']
            text = input_data_global_dict[node]['ocr']

            tensor_image = self.transforms(image)
            tensor_text = self.tokenizer.tokenize([text])[0]

            images.append(tensor_image)
            text_content.append(tensor_text)


        edges = []
        for edge in graphs.edges():
            node_in, node_out = edge
            edges.extend(
                        [
                         (nodes2idx_lut[node_in], nodes2idx_lut[node_out]),
                         (nodes2idx_lut[node_out], nodes2idx_lut[node_in])]
                         )

        queries = torch.stack([x for x in self.tokenizer.tokenize([sample['query'] for sample in batch])])

        return {
            'images': torch.stack(images),
            'textual_content': torch.stack(text_content),
            'raw_gt_text': [sample['ocr_gt'] for sample in batch],
            'raw_queries': [sample['query'] for sample in batch],
            'input_indices': input_nodes_idxs, # Input indices will be niizialized with a vision and text encoder,
            'edges': torch.tensor(edges),
            'queries': queries,
            'batch_indices': nodes_to_batch,
            'max_padding': max([len(sample['graph'].nodes()) for sample in batch]),
            # 'selected_idx': selected_nodes_idxs,
            'visual_queries': torch.stack([self.transforms(image['image']) for image in
                                           [sample['visual_query'] for sample in batch]]),
            'ocr_query': torch.stack([self.tokenizer.tokenize([image['ocr']])[0] for image in
                                           [sample['visual_query'] for sample in batch]])
        }





