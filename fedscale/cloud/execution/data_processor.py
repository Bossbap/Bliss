import torch
from torch.nn.utils.rnn import pad_sequence

# Import fllibs as a module so we always see the current tokenizer
import fedscale.cloud.fllibs as fllibs


def collate(examples):
    # Use runtime tokenizer from fllibs; fall back to default padding if missing
    tok = getattr(fllibs, "tokenizer", None)
    if tok is None or getattr(tok, "_pad_token", None) is None:
        return (pad_sequence(examples, batch_first=True), None)
    return (pad_sequence(examples, batch_first=True, padding_value=tok.pad_token_id), None)


def voice_collate_fn(batch):
    def func(p):
        return p[0].size(1)
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return (inputs, targets, input_percentages, target_sizes), None
