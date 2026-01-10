from datasets import load_dataset

def load_my_dataset(split="test", difficulty=None):
    """
    Load the APPS dataset from HuggingFace.

    Args:
        split: Which split to load ("train" or "test")
        difficulty: Filter by difficulty ("introductory", "interview", "competition"), or None for all

    Returns:
        Dataset object with APPS problems
    """
    dataset = load_dataset("codeparrot/apps", split=split)

    if difficulty:
        # Filter by difficulty level
        dataset = dataset.filter(lambda x: x.get('difficulty', '') == difficulty)

    return dataset
