from datasets import load_dataset


def load_triviaqa_sample():

    dataset = load_dataset(
        "trivia_qa",
        "rc"
    )

    sample = dataset["validation"][0]

    return {
        "question": sample["question"],
        "answer": sample["answer"]["value"],
        "context": sample["search_results"]["search_context"][0]
    }



def load_triviaqa_dataset():

    dataset = load_dataset(
        "trivia_qa",
        "rc"
    )

    return dataset["validation"]



if __name__ == "__main__":

    sample = load_triviaqa_sample()

    print(sample)