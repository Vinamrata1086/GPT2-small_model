# GPT-2 Small Model Implementation

This repository contains the Python and PyTorch code for implementing the GPT-2 small model (with 125 million parameters) from scratch, following the original paper and design. The code also includes a function to load the original GPT-2 125M model checkpoints and run a sample prediction.

## Model Description

The GPT-2 model is a transformer-based language model that can generate coherent and diverse text on various topics and domains. The model consists of the following components:

- Token and positional embeddings: The model uses byte-pair encoding (BPE) to tokenize the input text into subword units, and assigns each token an embedding vector. The model also uses learned positional embeddings to encode the relative position of each token in the input sequence.
- Transformer layers: The model uses 12 transformer layers, each consisting of a multi-head self-attention layer and a feed-forward network. The multi-head self-attention layer allows the model to attend to different parts of the input sequence, while the feed-forward network applies a non-linear transformation to the hidden state. The model also uses layer normalization and residual connections to facilitate the information flow and gradient propagation.
- Output layer: The model uses a linear layer with a softmax activation to produce the probability distribution over the vocabulary for each token. The model ties the weights of the output layer and the token embeddings to reduce the number of parameters.

## Code Structure

The code is organized as follows:

- `gpt2.py`: This file contains the main code for defining and instantiating the GPT-2 model class, as well as the transformer layer, the multi-head attention layer, and the feed-forward network classes. The file also contains a function to load the GPT-2 125M model checkpoints and a function to generate text using the model.
- `gpt2-pytorch_model.bin`: This file contains the pre-trained weights of the GPT-2 125M model.
- `README.md`: This file contains the documentation and instructions for using the code.

## Usage

To use the code, you need to have Python 3 and PyTorch installed. You also need to install the `transformers` library from HuggingFace to use the tokenizer. You can install it using `pip install transformers`.

To run the code, you can use the following commands:

- To instantiate the model and load the pre-trained weights, run `model = load_model(GPT2(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FFN_DIM, MAX_LEN, DROP_RATE), model_path)`, where `model_path` is the path to the `gpt2-pytorch_model.bin` file.
- To generate text using the model, run `output_text = generate_text(model, tokenizer, input_text, max_len, temperature, top_k, top_p)`, where `tokenizer` is the GPT-2 tokenizer from HuggingFace, `input_text` is the initial text to prompt the model, `max_len` is the maximum length of the generated text, `temperature` is the sampling temperature, `top_k` is the number of top candidates to sample from, and `top_p` is the cumulative probability threshold for nucleus sampling.

## References

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 9.
- Karpathy, A. (2020). nanoGPT. https://github.com/karpathy/nanoGPT
- makemore. (2020). How to build GPT-2 from scratch. https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&feature=shared
- Microsoft BING
