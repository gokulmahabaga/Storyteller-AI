pip install transformers torch

import torch
from transformers import pipeline, set_seed

class AIStoryteller:
    def __init__(self, model_name="gpt2"):
        """
        Initializes the AI Storyteller with a pre-trained language model.
        We use 'gpt2' as a common, relatively small model for demonstration.
        For better quality, consider 'gpt2-medium', 'gpt2-large', or even
        larger models if you have the computational resources.
        """
        print(f"Loading AI Storyteller model: {model_name}...")
        self.generator = pipeline('text-generation', model=model_name, device=0 if torch.cuda.is_available() else -1)
        # set_seed ensures reproducibility if you run with the same seed
        set_seed(42)
        print("Model loaded successfully!")

    def generate_story(self, prompt: str, max_length: int = 200, num_return_sequences: int = 1,
                       temperature: float = 0.9, top_k: int = 50, do_sample: bool = True) -> list:

        print(f"\nGenerating story with prompt: '{prompt}'...")
        generated_texts = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample
        )

        stories = [gen['generated_text'] for gen in generated_texts]
        return stories

    def get_input_from_user(self):
        """
        Handles user input for the story prompt and generation parameters.
        """
        print("\n--- AI Storyteller Input ---")
        prompt = input("Enter your story idea or starting sentence (e.g., 'A detective walked into a smoky bar'):\n> ")

        while True:
            try:
                max_length = int(input("Max story length (in words/tokens, e.g., 150-300 recommended for short stories): [200]\n> ") or "200")
                if max_length <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a positive integer for max length.")

        while True:
            try:
                num_stories = int(input("Number of stories to generate (e.g., 1-3): [1]\n> ") or "1")
                if num_stories <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a positive integer for number of stories.")

        while True:
            try:
                temperature = float(input("Creativity/Randomness (0.1 for focused, 1.0 for very creative): [0.9]\n> ") or "0.9")
                if not (0.1 <= temperature <= 1.5): # Reasonable range for temperature
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a float between 0.1 and 1.5.")

        return prompt, max_length, num_stories, temperature

    def display_output(self, stories: list):
        """
        Displays the generated stories.
        """
        print("\n--- Generated Stories ---")
        if not stories:
            print("No stories generated.")
            return

        for i, story_text in enumerate(stories):
            print(f"\nStory {i + 1}:\n")
            print(story_text)
            print("-" * 50) # Separator

# --- Main execution block ---
if __name__ == "__main__":
    # You can choose a different model if you have the resources.
    # e.g., "gpt2-medium", "distilgpt2" (smaller, faster)
    storyteller = AIStoryteller(model_name="gpt2")

    while True:
        prompt, max_length, num_stories, temperature = storyteller.get_input_from_user()

        if not prompt.strip():
            print("Prompt cannot be empty. Please provide a story idea.")
            continue

        generated_stories = storyteller.generate_story(
            prompt=prompt,
            max_length=max_length,
            num_return_sequences=num_stories,
            temperature=temperature
        )

        storyteller.display_output(generated_stories)

        again = input("\nGenerate another story? (yes/no): ").lower().strip()
        if again != 'yes':
            print("Thank you for using the AI Storyteller!")
            break

import time
import statistics
import re # For basic content checks
import torch
from transformers import pipeline, set_seed
# You would need libraries for more advanced metrics, e.g.,
# pip install textstat numpy
# import textstat
# import numpy as np

class AIStoryteller:
    def __init__(self, model_name="gpt2"):
        """
        Initializes the AI Storyteller with a pre-trained language model.
        We use 'gpt2' as a common, relatively small model for demonstration.
        For better quality, consider 'gpt2-medium', 'gpt2-large', or even
        larger models if you have the computational resources.
        """
        print(f"Loading AI Storyteller model: {model_name}...")
        self.generator = pipeline('text-generation', model=model_name, device=0 if torch.cuda.is_available() else -1)
        # set_seed ensures reproducibility if you run with the same seed
        set_seed(42)
        print("Model loaded successfully!")

    def generate_story(self, prompt: str, max_length: int = 200, num_return_sequences: int = 1,
                       temperature: float = 0.9, top_k: int = 50, do_sample: bool = True) -> list:
        """
        Generates a story based on a given prompt.

        Args:
            prompt (str): The starting prompt or idea for the story.
            max_length (int): The maximum number of tokens (words/subwords) in the generated story.
            num_return_sequences (int): How many different stories to generate for the given prompt.
            temperature (float): Controls the randomness of the generation. Higher = more creative/random.
                                 Lower = more focused/deterministic.
            top_k (int): Limits the sampling to the top_k most likely next tokens.
            do_sample (bool): If True, uses sampling for generation; otherwise, uses greedy decoding.

        Returns:
            list: A list of generated story texts.
        """
        print(f"\nGenerating story with prompt: '{prompt}'...")
        generated_texts = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample
        )

        stories = [gen['generated_text'] for gen in generated_texts]
        return stories

    def run_comprehensive_tests(self, test_scenarios: list):
        """
        Runs comprehensive tests including accuracy (quality proxies), edge cases,
        and measures response time.

        Args:
            test_scenarios (list): A list of dictionaries, each defining a test case.
                                   Example: {'prompt': '...', 'max_length': ..., 'genre': '...', 'expected_keywords': ['...']}
        """
        print("\n--- Running Comprehensive Tests ---")
        all_results = []
        response_times = []

        for i, scenario in enumerate(test_scenarios):
            print(f"\n--- Test Scenario {i + 1} ---")
            prompt = scenario.get('prompt', 'A story about a cat.')
            max_length = scenario.get('max_length', 150)
            num_return_sequences = scenario.get('num_return_sequences', 1)
            temperature = scenario.get('temperature', 0.9)
            expected_keywords = scenario.get('expected_keywords', [])
            expected_genre_keywords = scenario.get('expected_genre_keywords', [])
            expected_tone_keywords = scenario.get('expected_tone_keywords', [])
            test_type = scenario.get('type', 'standard') # e.g., 'standard', 'edge_case'


            print(f"Prompt: '{prompt}' (Type: {test_type})")
            print(f"Max Length: {max_length}, Temp: {temperature}")
            if expected_keywords:
                print(f"Expected Keywords: {expected_keywords}")
            if expected_genre_keywords:
                print(f"Expected Genre Keywords: {expected_genre_keywords}")
            if expected_tone_keywords:
                print(f"Expected Tone Keywords: {expected_tone_keywords}")


            start_time = time.perf_counter()
            generated_stories = self.generate_story(
                prompt=prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature
            )
            end_time = time.perf_counter()
            duration = end_time - start_time
            response_times.append(duration)

            print(f"Response Time: {duration:.4f} seconds")

            if not generated_stories:
                print("  FAILURE: No story generated.")
                all_results.append({'prompt': prompt, 'status': 'FAILED', 'reason': 'No story generated'})
                continue

            # Store results for human review and basic automated checks
            scenario_results = {
                'prompt': prompt,
                'stories': generated_stories,
                'response_time': duration,
                'automated_checks': {}
            }

            for story_idx, story_text in enumerate(generated_stories):
                print(f"\n--- Story {story_idx + 1} for prompt '{prompt[:50]}...' ---")
                print(story_text)

                # --- Automated 'Accuracy' Proxies / Basic Content Checks ---
                story_checks = {}
                words = story_text.split()
                word_count = len(words)
                story_checks['word_count'] = word_count
                story_checks['starts_with_prompt'] = story_text.lower().lstrip().startswith(prompt.lower().lstrip())

                # Keyword check (simple presence)
                missing_keywords = [kw for kw in expected_keywords if kw.lower() not in story_text.lower()]
                story_checks['missing_keywords'] = missing_keywords
                if missing_keywords:
                    print(f"  WARNING: Missing expected keywords: {missing_keywords}")

                # Genre/Tone keyword checks (simple presence)
                genre_hits = [kw for kw in expected_genre_keywords if kw.lower() in story_text.lower()]
                tone_hits = [kw for kw in expected_tone_keywords if kw.lower() in story_text.lower()]
                story_checks['genre_keyword_hits'] = genre_hits
                story_checks['tone_keyword_hits'] = tone_hits

                # Basic readability (requires textstat)
                # try:
                #     story_checks['flesch_reading_ease'] = textstat.flesch_reading_ease(story_text)
                # except Exception:
                #     story_checks['flesch_reading_ease'] = "N/A" # Handle empty strings or errors

                # You could add a simple check for repetition (e.g., using n-grams)
                # This is more complex and usually involves comparing generated segments

                scenario_results['automated_checks'][f'story_{story_idx+1}'] = story_checks
                print("-" * 60)

                # --- Human Evaluation Prompt (Crucial for true 'accuracy') ---
                print("\n--- Human Review for this Story ---")
                print("Rate the following (1=Poor, 5=Excellent):")
                coherence_rating = input("  Coherence & Flow: ") or '3'
                creativity_rating = input("  Creativity & Originality: ") or '3'
                relevance_rating = input("  Relevance to Prompt: ") or '3'
                consistency_rating = input("  Consistency (if applicable): ") or '3'
                overall_rating = input("  Overall Quality: ") or '3'
                qualitative_feedback = input("  Any specific comments?\n> ")

                scenario_results['human_feedback'] = {
                    'coherence': coherence_rating,
                    'creativity': creativity_rating,
                    'relevance': relevance_rating,
                    'consistency': consistency_rating,
                    'overall': overall_rating,
                    'comments': qualitative_feedback
                }

            all_results.append(scenario_results)
            print("\n" + "=" * 80) # Separator between scenarios

        self.display_test_summary(all_results, response_times)

    def display_test_summary(self, all_results: list, response_times: list):
        """Displays a summary of all tests."""
        print("\n\n--- Comprehensive Test Summary ---")
        print(f"Total Scenarios Tested: {len(all_results)}")

        # Response Time Summary
        if response_times:
            print("\nResponse Time Statistics:")
            print(f"  Min: {min(response_times):.4f} seconds")
            print(f"  Max: {max(response_times):.4f} seconds")
            print(f"  Average: {statistics.mean(response_times):.4f} seconds")
            print(f"  Median: {statistics.median(response_times):.4f} seconds")
            # print(f"  Standard Deviation: {statistics.stdev(response_times):.4f} seconds") # Requires more than one sample

        print("\nAutomated Checks Summary (first story of each scenario):")
        for res in all_results:
            status = res.get('status', 'N/A')
            if status == 'FAILED':
                print(f"  Prompt: '{res['prompt']}' - Status: FAILED ({res['reason']})")
                continue

            first_story_checks = res['automated_checks'].get('story_1', {})
            word_count = first_story_checks.get('word_count', 'N/A')
            starts_with = first_story_checks.get('starts_with_prompt', 'N/A')
            missing_kws = first_story_checks.get('missing_keywords', [])
            genre_hits = first_story_checks.get('genre_keyword_hits', [])
            tone_hits = first_story_checks.get('tone_keyword_hits', [])
            # flesch = first_story_checks.get('flesch_reading_ease', 'N/A')

            print(f"  Prompt: '{res['prompt'][:70]}...'")
            print(f"    Word Count: {word_count}")
            print(f"    Starts with Prompt: {starts_with}")
            print(f"    Missing Keywords: {missing_kws if missing_kws else 'None'}")
            print(f"    Genre Hits: {genre_hits if genre_hits else 'None'}")
            print(f"    Tone Hits: {tone_hits if tone_hits else 'None'}")
            # print(f"    Flesch Reading Ease: {flesch}")


        print("\n--- Human Feedback Averages (Overall Quality) ---")
        overall_ratings = []
        for res in all_results:
            if 'human_feedback' in res and 'overall' in res['human_feedback']:
                try:
                    overall_ratings.append(int(res['human_feedback']['overall']))
                except ValueError:
                    pass # Ignore invalid ratings

        if overall_ratings:
            print(f"  Average Overall Rating: {statistics.mean(overall_ratings):.2f}/5")
        else:
            print("  No valid human overall ratings collected.")

        print("\n--- All Raw Results (for detailed review) ---")
        # In a real system, you'd save this to a JSON or CSV file
        # import json
        # with open("test_results.json", "w") as f:
        #     json.dump(all_results, f, indent=2)
        # print("Detailed results saved to test_results.json")

# --- Main execution block ---
if __name__ == "__main__":
    storyteller = AIStoryteller(model_name="gpt2")

    # Define your test scenarios
    # Mix standard cases with edge cases
    test_scenarios = [
        # Standard Cases (testing general quality and adherence)
        {
            'prompt': "A young wizard discovers a secret passage in his school.",
            'max_length': 250,
            'temperature': 0.8,
            'expected_keywords': ['magic', 'passage', 'school', 'secret'],
            'expected_genre_keywords': ['fantasy', 'wizard'],
            'type': 'standard'
        },
        {
            'prompt': "The last detective in a neon-lit future city solves a bizarre case.",
            'max_length': 300,
            'temperature': 0.9,
            'expected_keywords': ['detective', 'city', 'bizarre'],
            'expected_genre_keywords': ['cyberpunk', 'sci-fi', 'mystery'],
            'expected_tone_keywords': ['gritty', 'noir'],
            'type': 'standard'
        },
        {
            'prompt': "A heartwarming story about an old woman and her garden.",
            'max_length': 200,
            'temperature': 0.7,
            'expected_keywords': ['garden', 'old woman', 'flowers'],
            'expected_tone_keywords': ['heartwarming', 'gentle'],
            'type': 'standard'
        },

        # Edge Cases
        {
            'prompt': "Silence fell. The cat", # Abrupt prompt, should continue smoothly
            'max_length': 100,
            'temperature': 0.9,
            'expected_keywords': ['cat', 'silence'], # Expect continuation
            'type': 'edge_case_abrupt_start'
        },
        {
            'prompt': "The man ate the rock. It was delicious. Then the rock", # Nonsensical premise
            'max_length': 100,
            'temperature': 0.9,
            'type': 'edge_case_nonsensical_premise'
        },
        {
            'prompt': "Repeat 'banana' fifty times: banana banana", # Repetition test
            'max_length': 150,
            'temperature': 0.1, # Lower temp to encourage direct repetition
            'expected_keywords': ['banana'],
            'type': 'edge_case_repetition_prompt'
        },
        {
            'prompt': "A story where the protagonist triumphs over adversity, but then immediately fails miserably.", # Contradictory outcome
            'max_length': 200,
            'temperature': 0.9,
            'expected_keywords': ['triumphs', 'adversity', 'fails miserably'],
            'type': 'edge_case_contradictory_plot'
        },
        {
            'prompt': "Generate a story about a dragon. Make sure the dragon is secretly a small, fluffy hamster.", # Absurd twist
            'max_length': 180,
            'temperature': 1.0,
            'expected_keywords': ['dragon', 'hamster', 'secretly'],
            'type': 'edge_case_absurd_twist'
        },
        {
            'prompt': "Why is the sky blue?", # Question prompt (should generate story, not answer fact)
            'max_length': 100,
            'temperature': 0.7,
            'type': 'edge_case_qa_prompt'
        },
        {
            'prompt': "A short, sad poem about a lonely cloud, but make it rhyme and have a happy ending.", # Conflicting constraints
            'max_length': 120,
            'temperature': 1.0,
            'expected_keywords': ['cloud', 'lonely', 'happy ending', 'rhyme'],
            'type': 'edge_case_conflicting_constraints'
        },
        {
            'prompt': "The year is 1984. Big Brother is watching. The protagonist, Winston, finds a small, rebellious book.", # Specific characters/settings
            'max_length': 250,
            'temperature': 0.8,
            'expected_keywords': ['1984', 'Big Brother', 'Winston', 'rebellious book'],
            'type': 'edge_case_specific_literary_reference'
        }
    ]

    storyteller.run_comprehensive_tests(test_scenarios)
