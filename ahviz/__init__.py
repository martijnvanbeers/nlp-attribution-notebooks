import numpy
import pandas
import torch

# make the dimension indices of the array explicit as
# a pandas dataframe MultiIndex
def create_indices(att:numpy.ndarray, names:list=['layer', 'head', 'sample', 'from_token', 'to_token'], **kwargs):
    """
    Build a pandas MultiIndex for a huggingface transformers attention matrix
    """
    spec = att.shape
    dims = {}
    for dim, size in reversed(list(enumerate(spec))):
        start = 0
        if names[dim] in kwargs:
            start = kwargs[names[dim]]
        if dim == len(spec) - 1:
            dims[dim] = numpy.arange(start, start + size) + 1
        else:
            for d in range(dim + 1, len(spec)):
                dims[d] = numpy.tile(dims[d], size)
            dims[dim] = numpy.repeat(numpy.arange(start, start + size) + 1, numpy.prod(spec[dim+1:]))

    return pandas.MultiIndex.from_arrays(
            list(dims.values()),
            names=reversed(names)
        )

def create_dataframe(att:torch.Tensor, ix:pandas.MultiIndex):
    df = pandas.DataFrame(
            att.flatten(), # turn the array into one long list of numbers
            columns=["attention_fraction"],
            index=ix, # indexed by its dimensions
        ).reset_index() # and then turn the dimensions into columns
    return df

def filter_mask(df:pandas.DataFrame, mask:torch.Tensor, how="both"):
    # filter out the masked tokens
    for sample in range(df['sample'].max()):
    #for sentence, toklist in enumerate(mask.tolist()):
#        # the next two lines filter out the first and last unmasked token which are [CLS] and [SEP] (for bert)
#        # comment them out to see the results with them included
#        final = max(np.nonzero(toklist)[0])
#        modified = [0] + toklist[1:final] + [0] + toklist[final+1:]

        for token in [i for i, v in enumerate(mask) if v == 0]:
            if how == "both":
                query = f"~(sample == {sample + 1} & (to_token == {token + 1} | from_token == {token + 1}))"
            else:
                query = f"~(sample == {sample + 1} & ({how}_token == {token + 1})"
            print(query)
            df = df.query(query)

