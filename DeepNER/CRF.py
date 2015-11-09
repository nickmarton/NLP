"""Conditional Random Field Model."""

def init_crf(X, y, verbose=False):
    """Initialize a CRF object."""
    trainer = pycrfsuite.Trainer(verbose=verbose)
    for xseq, yseq in zip(X, y):
        trainer.append(xseq, yseq)

    #set extra parameters of trainer object
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
    })

    trainer.train('model.crfsuite')

def main():
    """."""
    pass

if __name__ == "__main__":
    main()