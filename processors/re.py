import json
import os

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, subj_start=None, obj_start=None, subj_end=None, obj_end=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.subj_start = subj_start
        self.obj_start = obj_start
        self.subj_end = subj_end
        self.obj_end = obj_end

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, subj_start, obj_start):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.subj_start = subj_start
        self.obj_start = obj_start

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_json(self, file):
        with open(file, 'r', buffering=8*1024**2) as f:
            return json.load(f)


class TacredProcessor(DataProcessor):
    def __init__(self) -> None:
        super().__init__()
        # self.rel2id = {}
        self.relations = ['no_relation', 'org:founded_by', 'per:employee_of', 'org:alternate_names', 'per:cities_of_residence', 'per:children', 'per:title', 'per:siblings', 'per:religion', 'per:age', 'org:website', 'per:stateorprovinces_of_residence', 'org:member_of', 'org:top_members/employees', 'per:countries_of_residence', 'org:city_of_headquarters', 'org:members', 'org:country_of_headquarters', 'per:spouse', 'org:stateorprovince_of_headquarters', 'org:number_of_employees/members', 'org:parents', 'org:subsidiaries', 'per:origin', 'org:political/religious_affiliation', 'per:other_family', 'per:stateorprovince_of_birth', 'org:dissolved', 'per:date_of_death', 'org:shareholders', 'per:alternate_names', 'per:parents', 'per:schools_attended', 'per:cause_of_death', 'per:city_of_death', 'per:stateorprovince_of_death', 'org:founded', 'per:country_of_birth', 'per:date_of_birth', 'per:city_of_birth', 'per:charges', 'per:country_of_death']
        # self.rel2id = {label: i for i, label in enumerate(self.relations)}

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, 'train.json')), 'train')

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(self._read_json(os.path.join(data_dir, 'dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(self._read_json(os.path.join(data_dir, 'test.json')), 'test')

    def get_labels(self):
        return self.relations

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line['id']
            text_a = ' '.join(line['token'])
            text_b = None
            # if line['relation'] not in self.rel2id:
            #     self.rel2id[line['relation']] = len(self.rel2id)
            # label = self.rel2id[line['relation']]
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             label=line['relation'],
                             subj_start=line['subj_start'],
                             obj_start=line['obj_start'],
                             subj_end=line['subj_end'],
                             obj_end=line['obj_end']))
        return examples


class ReTacredProcessor(TacredProcessor):
    def __init__(self) -> None:
        super().__init__()
        # self.rel2id = {}
        self.relations = ['no_relation', 'org:members', 'per:siblings', 'per:spouse', 'org:country_of_branch', 'per:country_of_death', 'per:parents', 'per:stateorprovinces_of_residence', 'org:top_members/employees', 'org:dissolved', 'org:number_of_employees/members', 'per:stateorprovince_of_death', 'per:origin', 'per:children', 'org:political/religious_affiliation', 'per:city_of_birth', 'per:title', 'org:shareholders', 'per:employee_of', 'org:member_of', 'org:founded_by', 'per:countries_of_residence', 'per:other_family', 'per:religion', 'per:identity', 'per:date_of_birth', 'org:city_of_branch', 'org:alternate_names', 'org:website', 'per:cause_of_death', 'org:stateorprovince_of_branch', 'per:schools_attended', 'per:country_of_birth', 'per:date_of_death', 'per:city_of_death', 'org:founded', 'per:cities_of_residence', 'per:age', 'per:charges', 'per:stateorprovince_of_birth']

class SemevalProcessor(TacredProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.relations = ['Other', 'Component-Whole(e2,e1)', 'Instrument-Agency(e2,e1)', 'Member-Collection(e1,e2)', 'Cause-Effect(e2,e1)', 'Entity-Destination(e1,e2)', 'Content-Container(e1,e2)', 'Message-Topic(e1,e2)', 'Product-Producer(e2,e1)', 'Member-Collection(e2,e1)', 'Entity-Origin(e1,e2)', 'Cause-Effect(e1,e2)', 'Component-Whole(e1,e2)', 'Message-Topic(e2,e1)', 'Product-Producer(e1,e2)', 'Entity-Origin(e2,e1)', 'Content-Container(e2,e1)', 'Instrument-Agency(e1,e2)', 'Entity-Destination(e2,e1)']


class TacrevProcessor(TacredProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.relations = ['no_relation', 'org:founded', 'org:subsidiaries', 'per:date_of_birth', 'per:cause_of_death', 'per:age', 'per:stateorprovince_of_birth', 'per:countries_of_residence', 'per:country_of_birth', 'per:stateorprovinces_of_residence', 'org:website', 'per:cities_of_residence', 'per:parents', 'per:employee_of', 'per:city_of_birth', 'org:parents', 'org:political/religious_affiliation', 'per:schools_attended', 'per:country_of_death', 'per:children', 'org:top_members/employees', 'per:date_of_death', 'org:members', 'org:alternate_names', 'per:religion', 'org:member_of', 'org:city_of_headquarters', 'per:origin', 'org:shareholders', 'per:charges', 'per:title', 'org:number_of_employees/members', 'org:dissolved', 'org:country_of_headquarters', 'per:alternate_names', 'per:siblings', 'org:stateorprovince_of_headquarters', 'per:spouse', 'per:other_family', 'per:city_of_death', 'per:stateorprovince_of_death', 'org:founded_by']


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # tokens_a = tokenizer.tokenize(example.text_a)
        example.text_a = example.text_a.split(' ')
        subj = tokenizer.tokenize(' '.join(example.text_a[example.subj_start:example.subj_end+1]))
        obj = tokenizer.tokenize(' '.join(example.text_a[example.obj_start:example.obj_end+1]))
        if example.subj_start < example.obj_start:
            left = tokenizer.tokenize(' '.join(example.text_a[:example.subj_start]))
            mid = tokenizer.tokenize(' '.join(example.text_a[example.subj_end+1:example.obj_start]))
            right = tokenizer.tokenize(' '.join(example.text_a[example.obj_end+1:]))
            tokens_a = left + subj + mid + obj + right
            subj_start = len(left) + 1
            obj_start = len(left) + len(mid) + len(subj) + 1
        else:
            left = tokenizer.tokenize(' '.join(example.text_a[:example.obj_start]))
            mid = tokenizer.tokenize(' '.join(example.text_a[example.obj_end+1:example.subj_start]))
            right = tokenizer.tokenize(' '.join(example.text_a[example.subj_end+1:]))
            tokens_a = left + obj + mid + subj + right
            obj_start = len(left) + 1
            subj_start = len(left) + len(mid) + len(obj) + 1
        if subj_start >= max_seq_length - 1:
            subj_start = 0
        if obj_start >= max_seq_length - 1:
            obj_start = 0
        tokens_b = None
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # cnt = 0
        # for i, item in enumerate(tokens):
        #     if item == '<E1>':
        #         obj_start = i
        #         cnt += 1
        #     elif item == '<E2>':
        #         subj_start = i
        #         cnt += 1
        #     if cnt == 2:
        #         break
        

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        # label_id = example.label

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          subj_start=subj_start,
                          obj_start=obj_start))
    return features, label_map

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

PROCESSORS = {
    'tacred': TacredProcessor,
    'retacred': ReTacredProcessor,
    'semeval': SemevalProcessor,
    'tacrev': TacrevProcessor
}