# Storyteller-AI

OUTPUT:
Loading AI Storyteller model: gpt2...
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
config.json: 100%
 665/665 [00:00<00:00, 42.6kB/s]
model.safetensors: 100%
 548M/548M [00:13<00:00, 46.7MB/s]
generation_config.json: 100%
 124/124 [00:00<00:00, 9.91kB/s]
tokenizer_config.json: 100%
 26.0/26.0 [00:00<00:00, 1.48kB/s]
vocab.json: 100%
 1.04M/1.04M [00:00<00:00, 4.46MB/s]
merges.txt: 100%
 456k/456k [00:00<00:00, 9.72MB/s]
tokenizer.json: 100%
 1.36M/1.36M [00:00<00:00, 18.5MB/s]
Device set to use cpu
Model loaded successfully!

--- AI Storyteller Input ---
Enter your story idea or starting sentence (e.g., 'A detective walked into a smoky bar'):
> A detective walked into a smoky bar
Max story length (in words/tokens, e.g., 150-300 recommended for short stories): [200]
> 200
Number of stories to generate (e.g., 1-3): [1]
> 1
Creativity/Randomness (0.1 for focused, 1.0 for very creative): [0.9]
> 1.0
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

Generating story with prompt: 'A detective walked into a smoky bar'...

--- Generated Stories ---

Story 1:

A detective walked into a smoky bar, dressed in black. He looked up to see a white man in front with a black, white gun. "I told him I'm going to do your business," the man replied, "I don't need your help."

We are not being completely honest here. He was a good guy.

On the morning of February 21, 1981, I was out to dinner. As this incident came to a head in California and we both saw people shooting at each other and with knives.

Here in San Diego County, our town and a local radio station, KQED, had a small group with black people to broadcast a radio show to show the people. I know exactly who that person was: a black man who had been shot in the back of the neck and was sitting in the car with his wife and his 4-month-old, his 6-month-old daughter in custody. His wife and the child were gone.

They could not see him, but in the window, there was a large group of people in a car. One of them was holding a knife. My husband and I took the knife to the throat, and we cut him three to four times but it didn't kill him.

I knew
--------------------------------------------------

Generate another story? (yes/no): no
Thank you for using the AI Storyteller!

TESTING:
Loading AI Storyteller model: gpt2...
Device set to use cpu
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Both `max_new_tokens` (=256) and `max_length`(=250) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
Model loaded successfully!

--- Running Comprehensive Tests ---

--- Test Scenario 1 ---
Prompt: 'A young wizard discovers a secret passage in his school.' (Type: standard)
Max Length: 250, Temp: 0.8
Expected Keywords: ['magic', 'passage', 'school', 'secret']
Expected Genre Keywords: ['fantasy', 'wizard']

Generating story with prompt: 'A young wizard discovers a secret passage in his school.'...
Response Time: 17.5086 seconds

--- Story 1 for prompt 'A young wizard discovers a secret passage in his s...' ---
A young wizard discovers a secret passage in his school. The wizard begins to learn that the Chamber of Secrets is a secret, and that it is a secret that the wizard must obey to escape, so he has to discover the secret. The Wizard of Oz lives by his oath to never kill except for the bestial evil of his enemies.

A young wizard discovers a secret passage in his school. The wizard begins to learn that the Chamber of Secrets is a secret, and that it is a secret that the wizard must obey to escape, so he has to discover the secret. The Wizard of Oz lives by his oath to never kill except for the bestial evil of his enemies.

The Magical Girl, a schoolgirl who becomes one of the first witches and wizards to appear in the film, is introduced in Harry Potter and the Chamber of Secrets. She is the main antagonist and the only female wizard of the Hogwarts castle.

The Magical Girl is introduced in Harry Potter and the Chamber of Secrets. She is the main antagonist and the only female wizard of the Hogwarts castle.

The Magical Girl is named after the character of Mary Magdalene Dumbledore, who became the first witch to appear in the film as a young magical girl.

Beware of Wizarding Death.

The most common curse
------------------------------------------------------------

--- Human Review for this Story ---
Rate the following (1=Poor, 5=Excellent):
  Coherence & Flow: 2
  Creativity & Originality: 2
  Relevance to Prompt: 4
  Consistency (if applicable): 2
  Overall Quality: 3
  Any specific comments?
> repetitive
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Both `max_new_tokens` (=256) and `max_length`(=300) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
(This contiues upto indented times)
