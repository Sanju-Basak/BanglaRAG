if __name__ == "__main__":
    import numpy as np
    from ragatouille import RAGTrainer
    trainer = RAGTrainer(model_name="l3cube_pune", pretrained_model_name="l3cube-pune/bengali-sentence-similarity-sbert", language_code="bn")
    # import random
    from datasets import load_dataset
    dataset= load_dataset("csebuetnlp/squad_bn", split="train")
    # dataset = dataset.select(range(200))
    from normalizer import normalize
    dataset = dataset.map(lambda x: {"context": normalize(x["context"]), "question": normalize(x["question"])})
    collection= dataset["context"]

    pairs= []
    pairs = list(zip(dataset["question"], dataset["context"]))
    # %%
    

    # %% [markdown]
    # Here, we have created pairs.It's common for retrieval training data to be stored in a lot of different ways: pairs of [query, positive], pairs of [query, passage, label], triplets of [query, positive, negative], or triplets of [query, list_of_positives, list_of_negatives]. No matter which format your data's in, you don't need to worry about it: RAGatouille will generate ColBERT-friendly triplets for you, and export them to disk for easy `dvc` or `wandb` data tracking.
    # 
    # Speaking of, let's process the data so it's ready for training. `RAGTrainer` has a `prepare_training_data` function, which will perform all the necessary steps. One of the steps it performs is called **hard negative mining**: that's searching the full collection of documents (even those not linked to a query) to retrieve passages that are semantically close to a query, but aren't actually relevant. Using those to train retrieval models has repeatedly been shown to greatly improve their ability to find actually relevant documents, so it's a very important step! 
    # 
    # RAGatouille handles all of this for you. By default, it'll fetch 10 negative examples per query, but you can customise this with `num_new_negatives`. You can also choose not to mine negatives and just sample random examples to speed up things, this might lower performance but will run done much quicker on large volumes of data, just set `mine_hard_negatives` to `False`. If you've already mined negatives yourself, you can set `num_new_negatives` to 0 to bypass this entirely.

    # %%
    trainer.prepare_training_data(raw_data=pairs, data_out_path="./data/", all_documents=collection, num_new_negatives=20, mine_hard_negatives=True)

    # %% [markdown]
    # Our training data's now fully processed and saved to disk in `data_out_path`! We're now ready to begin training our model with the `train` function. `train` takes many arguments, but the set of default is already fairly strong!
    # 
    # Don't be surprised you don't see an `epochs` parameter here, ColBERT will train until it either reaches `maxsteps` or has seen the entire training data once (a full epoch), this is by design!

    # %%

    trainer.train(batch_size=32,
                nbits=4, # How many bits will the trained model use when compressing indexes
                maxsteps=500000, # Maximum steps hard stop
                use_ib_negatives=True, # Use in-batch negative to calculate loss
                dim=128, # How many dimensions per embedding. 128 is the default and works well.
                learning_rate=3e-5, # Learning rate, small values ([3e-6,3e-5] work best if the base model is BERT-like, 5e-6 is often the sweet spot)
                doc_maxlen=256, # Maximum document length. Because of how ColBERT works, smaller chunks (128-256) work very well.
                use_relu=False, # Disable ReLU -- doesn't improve performance
                warmup_steps="auto", # Defaults to 10%
                accumsteps=1
                )


    # %%


    # %% [markdown]
    # And you're now done training! Your model is saved at the path it outputs, with the final checkpoint always being in the `.../checkpoints/colbert` path, and intermediate checkpoints saved at `.../checkpoints/colbert-{N_STEPS}`.
    # 
    # You can now use your model by pointing at its local path, or upload it to the huggingface hub to share it with the world!


