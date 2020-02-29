import torch
import numpy
from torchtext.data.metrics import bleu_score


def itos(idx_list, TRG):
    sentence = [TRG.vocab.itos[idx] for idx in idx_list]
    return sentence

def count_bleu(output, trg, TRG):
    # output shape: seq_len * batch_size * feature 
    # trg shape: seq_len * batch_size
    # corpus level or sentence level bleu ?
    output = output.permute(1,0,2).max(2)[1]
    trg = trg.permute(1,0)
    candidate_corpus = [itos(idx_list, TRG) for idx_list in output]
    references_corpus = [[itos(idx_list, TRG)] for idx_list in trg]
    return bleu_score(candidate_corpus, references_corpus)

def count_wer(output, tgt):
    """
      output: torch Tensor of shape T x E
      tgt: torch Tensor of shape 
    """
    output = torch.argmax(output,1)
    output = output.detach().data.cpu().numpy()
    tgt = tgt.detach().data.cpu().numpy()
    return wer(tgt,output)

def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : reference list
    h : hypothese list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]/len(r)

