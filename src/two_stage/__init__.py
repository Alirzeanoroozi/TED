r"""Two-stage TEDPred: frozen ESM2 backbone + per-residue/per-pair domain
segmentation (Stage A) + hierarchical CATH classification (Stage B).

This package replaces the character-level seq2seq decoder (src/model.py) with a
decomposed architecture:

    sequence --ESM2--> per-residue embeddings
        |--> Stage A: residue (domain/linker) + pairwise co-membership heads
        |             -> cluster into domains (Chainsaw-style)
        \--> Stage B: pool each domain's embedding -> 4 hierarchical C/A/T/H heads

At inference the predicted domains + CATH labels are serialized back to the
canonical ``chopping_star`` string so the existing benchmark/eval pipeline
(``benchmark/ted_eval.py``) works unchanged.
"""
