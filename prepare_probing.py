from collections import namedtuple, defaultdict
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from argparse import ArgumentParser
import os
import h5py
import numpy as np
import dill
from manifolds import Lorentz

from modeling import BertConfig, BertModel, BertPreTrainedModel
from tokenization import BertTokenizer
from frechetmean import frechet_mean
from frechetmean import Lorentz as LorentzFrechet

parser = ArgumentParser()
parser.add_argument('--data_path', default='/home/chenweize/hilbert/data/download/treebank_3', type=str)
parser.add_argument('--init_checkpoint', default='./results/new_base_dp=0.1_attdp=0.1_lr=2e-3_step=30000/checkpoints/ckpt_33000.pt', type=str)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--save_path', default='/home/chenweize/hilbert/data/download/treebank_3/embedding/')
parser.add_argument('--config_file', default='bert_base.json')
parser.add_argument('--vocab_file', default='vocab/vocab')
args = parser.parse_args()


def get_observation_class(fieldnames):
    """
    Returns a namedtuple class for a single observation.

    The namedtuple class is constructed to hold all language and annotation
    information for a single sentence or document.

    Args:
        fieldnames: a list of strings corresponding to the information in each
            row of the conllx file being read in. (The file should not have
            explicit column headers though.)
    Returns:
        A namedtuple class; each observation in the dataset will be an instance
        of this class.
    """
    return namedtuple("Observation", fieldnames)

def load_text(data_path):
    """
    Yields batches of lines describing a sentence in conllx.

    Args:
        data_path: Path of a conllx file.
    Yields:
        a list of lines describing a single sentence in conllx.
    """
    text, t = [], []

    for line in tqdm(open(data_path)):
        if line.startswith("#"):
            continue

        if not line.strip():
            text += [" ".join(t)]
            t = []
        else:
            t.append(line.split("\t")[1])

    return text

def generate_lines_for_sent(lines):
    """
    Yields batches of lines describing a sentence in conllx.

    Args:
        lines: Each line of a conllx file.
    Yields:
        a list of lines describing a single sentence in conllx.
    """

    buf = []
    for line in lines:
        if line.startswith("#"):
            continue
        if not line.strip():
            if buf:
                yield buf
                buf = []
            else:
                continue
        else:
            buf.append(line.strip())
    if buf:
        yield buf

def load_conll_dataset(filepath, observation_class):
    """
    Reads in a conllx file; generates Observation objects

    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
        filepath: the filesystem path to the conll dataset

    Returns:
        A list of Observations
    """
    observations = []

    lines = open(filepath).readlines()
    for buf in generate_lines_for_sent(lines):
        conllx_lines = []
        for line in buf:
            conllx_lines.append(line.strip().lower().split("\t"))
        embeddings = [None for x in range(len(conllx_lines))]
        observation = observation_class(*zip(*conllx_lines), embeddings)
        observations.append(observation)

    return observations


def get_dataloader(tokenizer, text, batch_size):
    input_ids_tensors = []
    input_mask_tensors = []
    segment_ids_tensors = []
    max_len = 0
    for index, line in tqdm(enumerate(text)):
        line = line.strip()  # Remove trailing characters
        line = tokenizer.tokenize(line)
        line = ["[CLS]"] + line + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(line)
        if len(input_ids) > max_len:
            max_len = len(input_ids)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        padding = [0] * (168 - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        input_ids_tensors.append(input_ids)
        input_mask_tensors.append(input_mask)
        segment_ids_tensors.append(segment_ids)
    input_ids_tensors = torch.tensor(input_ids_tensors, dtype=torch.long)
    input_mask_tensors = torch.tensor(input_mask_tensors, dtype=torch.long)
    segment_ids_tensors = torch.tensor(segment_ids_tensors, dtype=torch.long)
    return DataLoader(TensorDataset(input_ids_tensors, input_mask_tensors, segment_ids_tensors), batch_size=batch_size, shuffle=False)

# def save_embedding(model, dataloader, save_file):
#     index = 0
#     with torch.no_grad():
#         model.eval()
#         with h5py.File(save_file, 'w') as f:
#             for batch in tqdm(dataloader, total=len(dataloader)):
#                 batch = tuple(t.cuda() for t in batch)
#                 input_ids, input_mask, segment_ids = batch
#                 layer_logits, _ = model(input_ids, segment_ids, input_mask)
#                 tmp = torch.stack([layer_logits[j][0][:input_mask[0].sum()] for j in range(len(layer_logits))], dim=0).cpu().numpy()
#                 for i in range(len(input_ids)):
#                     f.create_dataset(str(index), data=torch.stack([layer_logits[j][i][:input_mask[i].sum()] for j in range(len(layer_logits))], dim=0).cpu().numpy())
#                     index += 1


def save_embedding(path, text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT):
    """
    Takes raw text and saves BERT-cased features for that text to disk
    Adapted from the BERT readme (and using the corresponding package) at
    https://github.com/huggingface/pytorch-pretrained-BERT
    """

    model.eval()
    with h5py.File(path, "w") as fout:
        for index, line in tqdm(enumerate(text)):
            line = line.strip()  # Remove trailing characters
            line = "[CLS] " + line.lower() + " [SEP]"
            tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segment_ids = [0] * len(tokenized_text)
            input_mask = [1] * len(tokenized_text)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).cuda().long()
            segments_tensors = torch.tensor([segment_ids]).cuda().long()
            mask_tensors = torch.tensor([input_mask]).cuda().long()

            with torch.no_grad():
                encoded_layers, _ = model(
                    tokens_tensor, segments_tensors, mask_tensors
                )
                # embeddings + 12 layers
                # encoded_layers = encoded_layers
            dset = fout.create_dataset(
                str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT)
            )
            dset[:, :, :] = np.vstack([x.cpu().numpy() for x in encoded_layers])

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    """
    Aligns tokenized and untokenized sentence given subwords "##" prefixed

    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.

    Args:
        tokenized_sent: a list of strings describing a subword-tokenized sentence
        untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
        A dictionary of type {int: list(int)} mapping each untokenized sentence
        index to a list of subword-tokenized sentence indices
    """
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(
        tokenized_sent
    ):

        while tokenized_sent_index + 1 < len(tokenized_sent) and tokenized_sent[
            tokenized_sent_index + 1
        ].startswith("##"):

            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1

        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1

    return mapping


def embed_bert_observation(hdf5_path, observations, tokenizer, observation_class, layer_index):
    hf = h5py.File(hdf5_path, "r")
    indices = list(hf.keys())

    single_layer_features_list = []
    manifold = Lorentz()
    for index in tqdm(sorted([int(x) for x in indices]), desc="[aligning embeddings]"):
        observation = observations[index]
        feature_stack = hf[str(index)]
        single_layer_features = torch.from_numpy(feature_stack[layer_index])
        tokenized_sent = ["[CLS]"] + tokenizer.wordpiece_tokenizer.tokenize(" ".join(observation.sentence)) + ["[SEP]"]
        untokenized_sent = observation.sentence
        untok_tok_mapping = match_tokenized_to_untokenized(
            tokenized_sent, untokenized_sent
        )
        assert single_layer_features.shape[0] == len(tokenized_sent)
        single_layer_features = torch.cat(
            [
                # manifold.elementwise_mid_point(
                    single_layer_features[
                        untok_tok_mapping[i][0] : untok_tok_mapping[i][0] + 1, :
                    ]
                # )
                if len(untok_tok_mapping[i]) == 1 else 
                    frechet_mean(single_layer_features[
                        untok_tok_mapping[i][0] : untok_tok_mapping[i][-1] + 1, :
                    ], LorentzFrechet()).unsqueeze(0) for i in range(len(untokenized_sent))
            ],
            dim=0
        )
        assert single_layer_features.shape[0] == len(observation.sentence)
        single_layer_features_list.append(single_layer_features)

    embeddings = single_layer_features_list
    embedded_observations = []
    for observation, embedding in zip(observations, embeddings):
        embedded_observation = observation_class(*(observation[:-1]), embedding)
        embedded_observations.append(embedded_observation)

    return embedded_observations


class BertForLoad(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForLoad, self).__init__(config)
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(input_ids, token_type_ids, attention_mask)


class Task:
    """Abstract class representing a linguistic task mapping texts to labels."""

    @staticmethod
    def labels(observation):
        """Maps an observation to a matrix of labels.
    
    Should be overriden in implementing classes.
    """
        raise NotImplementedError


class ParseDistanceTask(Task):
    """Maps observations to dependency parse distances between words."""

    @staticmethod
    def labels(observation):
        """Computes the distances between all pairs of words; returns them as a torch tensor.

    Args:
        observation: a single Observation class for a sentence:
    Returns:
        A torch tensor of shape (sentence_length, sentence_length) of distances
        in the parse tree as specified by the observation annotation.
    """
        sentence_length = len(
            observation[0]
        )  # All observation fields must be of same length
        distances = torch.zeros((sentence_length, sentence_length))
        for i in range(sentence_length):
            for j in range(i, sentence_length):
                i_j_distance = ParseDistanceTask.distance_between_pairs(
                    observation, i, j
                )
                distances[i][j] = i_j_distance
                distances[j][i] = i_j_distance
        return distances

    @staticmethod
    def distance_between_pairs(observation, i, j, head_indices=None):
        """Computes path distance between a pair of words

    TODO: It would be (much) more efficient to compute all pairs' distances at once;
        this pair-by-pair method is an artefact of an older design, but
        was unit-tested for correctness... 

    Args:
        observation: an Observation namedtuple, with a head_indices field.
            or None, if head_indies != None
        i: one of the two words to compute the distance between.
        j: one of the two words to compute the distance between.
        head_indices: the head indices (according to a dependency parse) of all
            words, or None, if observation != None.

    Returns:
        The integer distance d_path(i,j)
    """
        if i == j:
            return 0
        if observation:
            head_indices = []
            number_of_underscores = 0
            for elt in observation.head_indices:
                if elt == "_":
                    head_indices.append(0)
                    number_of_underscores += 1
                else:
                    head_indices.append(int(elt) + number_of_underscores)
        i_path = [i + 1]
        j_path = [j + 1]
        i_head = i + 1
        j_head = j + 1
        while True:
            if not (i_head == 0 and (i_path == [i + 1] or i_path[-1] == 0)):
                i_head = head_indices[i_head - 1]
                i_path.append(i_head)
            if not (j_head == 0 and (j_path == [j + 1] or j_path[-1] == 0)):
                j_head = head_indices[j_head - 1]
                j_path.append(j_head)
            if i_head in j_path:
                j_path_length = j_path.index(i_head)
                i_path_length = len(i_path) - 1
                break
            elif j_head in i_path:
                i_path_length = i_path.index(j_head)
                j_path_length = len(j_path) - 1
                break
            elif i_head == j_head:
                i_path_length = len(i_path) - 1
                j_path_length = len(j_path) - 1
                break
        total_length = j_path_length + i_path_length
        return total_length


class ParseDepthTask:
    """Maps observations to a depth in the parse tree for each word"""

    @staticmethod
    def labels(observation):
        """Computes the depth of each word; returns them as a torch tensor.

    Args:
        observation: a single Observation class for a sentence:
    Returns:
        A torch tensor of shape (sentence_length,) of depths
        in the parse tree as specified by the observation annotation.
    """
        sentence_length = len(
            observation[0]
        )  # All observation fields must be of same length
        depths = torch.zeros(sentence_length)
        for i in range(sentence_length):
            depths[i] = ParseDepthTask.get_ordering_index(observation, i)
        return depths

    @staticmethod
    def get_ordering_index(observation, i, head_indices=None):
        """Computes tree depth for a single word in a sentence

    Args:
        observation: an Observation namedtuple, with a head_indices field.
            or None, if head_indies != None
        i: the word in the sentence to compute the depth of
        head_indices: the head indices (according to a dependency parse) of all
            words, or None, if observation != None.

    Returns:
        The integer depth in the tree of word i
    """
        if observation:
            head_indices = []
            number_of_underscores = 0
            for elt in observation.head_indices:
                if elt == "_":
                    head_indices.append(0)
                    number_of_underscores += 1
                else:
                    head_indices.append(int(elt) + number_of_underscores)
        length = 0
        i_head = i + 1
        while True:
            i_head = head_indices[i_head - 1]
            if i_head != 0:
                length += 1
            else:
                return length

class ObservationIterator(Dataset):
    """
    List Container for lists of Observations and labels for them.
    Used as the iterator for a PyTorch dataloader.
    -----
    author: @john-hewitt
    https://github.com/john-hewitt/structural-probes
    """

    def __init__(self, observations, task):
        self.observations = observations
        self.set_labels(observations, task)

    def set_labels(self, observations, task):
        """
        Constructs aand stores label for each observation.

        Args:
            observations: A list of observations describing a dataset
            task: a Task object which takes Observations and constructs labels.
        """
        self.labels = []
        for observation in tqdm(observations, desc="[computing labels]"):
            self.labels.append(task.labels(observation))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]


config = BertConfig.from_json_file(args.config_file)
# Padding for divisibility by 8
if config.vocab_size % 8 != 0:
    config.vocab_size += 8 - (config.vocab_size % 8)
config.output_all_encoded_layers = True
model = BertForLoad(config)
model.load_state_dict(
    torch.load(args.init_checkpoint, map_location='cpu')["model"],
    strict=False,
)
model.cuda()
tokenizer = BertTokenizer(
    args.vocab_file,
    do_lower_case=True,
    max_len=512,
)  # for bert large

train_data_path = os.path.join(args.data_path, "ptb3-wsj-train.conllx")
dev_data_path = os.path.join(args.data_path, "ptb3-wsj-dev.conllx")
test_data_path = os.path.join(args.data_path, "ptb3-wsj-test.conllx")
train_text = load_text(train_data_path)
dev_text = load_text(dev_data_path)
test_text = load_text(test_data_path)

train_dataloader = get_dataloader(tokenizer, train_text, args.batch_size)
dev_dataloader = get_dataloader(tokenizer, dev_text, args.batch_size)
test_dataloader = get_dataloader(tokenizer, test_text, args.batch_size)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

train_hdf5_path = os.path.join(args.save_path, 'train.hdf5')
dev_hdf5_path = os.path.join(args.save_path, 'dev.hdf5')
test_hdf5_path = os.path.join(args.save_path, 'test.hdf5')
LAYER_COUNT = 12
FEATURE_COUNT = 768

# save_embedding(model, train_dataloader, train_hdf5_path)
# save_embedding(model, dev_dataloader, dev_hdf5_path)
# save_embedding(model, test_dataloader, test_hdf5_path)
# save_embedding(train_hdf5_path, train_text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT)
# save_embedding(dev_hdf5_path, dev_text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT)
# save_embedding(test_hdf5_path, test_text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT)

observation_fieldnames = [
    "index",
    "sentence",
    "lemma_sentence",
    "upos_sentence",
    "xpos_sentence",
    "morph",
    "head_indices",
    "governance_relations",
    "secondary_relations",
    "extra_info",
    "embeddings",
]

observation_class = get_observation_class(observation_fieldnames)

train_observations = load_conll_dataset(train_data_path, observation_class)
dev_observations = load_conll_dataset(dev_data_path, observation_class)
test_observations = load_conll_dataset(test_data_path, observation_class)

# for task_name in ['distance', 'depth']:
#     if task_name == "distance":
#         task = ParseDistanceTask()
#     else:
#         task = ParseDepthTask()
task_name = 'distance'
task = ParseDistanceTask()
# task_name = 'depth'
# task = ParseDepthTask()
for layer_index in range(config.num_hidden_layers):
    train_observations = embed_bert_observation(train_hdf5_path, train_observations, tokenizer, observation_class, layer_index)
    dev_observations = embed_bert_observation(dev_hdf5_path, dev_observations, tokenizer, observation_class, layer_index)
    test_observations = embed_bert_observation(test_hdf5_path, test_observations, tokenizer, observation_class, layer_index)
    train_dataset = ObservationIterator(train_observations, task)
    dev_dataset = ObservationIterator(dev_observations, task)
    test_dataset = ObservationIterator(test_observations, task)
    torch.save(train_dataset, os.path.join(args.save_path, task_name, 'train_{}.pt'.format(layer_index)), pickle_module=dill)
    torch.save(dev_dataset, os.path.join(args.save_path, task_name, 'dev_{}.pt'.format(layer_index)), pickle_module=dill)
    torch.save(test_dataset, os.path.join(args.save_path, task_name, 'test_{}.pt'.format(layer_index)), pickle_module=dill)
