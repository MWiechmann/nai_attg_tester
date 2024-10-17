# Standard library imports
import configparser
import os
import ast
import asyncio
import time
import logging
from datetime import datetime
from pathlib import Path

# Third-party imports
import pandas as pd
from novelai_api.NovelAI_API import NovelAIAPI
from novelai_api.Preset import Model, Preset
from novelai_api.GlobalSettings import GlobalSettings
from novelai_api.Tokenizer import Tokenizer
from novelai_api.utils import b64_to_tokens, tokens_to_b64

# Define constants
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / 'config'

def parse_config_value(value):
    """
    Parse a config value, preserving spaces if it's a quoted string.
    """
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        return ast.literal_eval(value)
    return value

# Read Settings
config_file = CONFIG_DIR / 'genre_clio_settings.ini'
config = configparser.ConfigParser()
config.read(config_file)

# Access the General Settings
run_name = config['GENERAL']['run_name']
auth_method = config['GENERAL']['auth_method']

# Access Story Generation Settings
delay_time = int(config['STORY GENERATION - GEN SETTINGS']['delay_time'])
generation_timeout = int(config['STORY GENERATION - GEN SETTINGS']['generation_timeout'])
max_failed_gens = int(config['STORY GENERATION - GEN SETTINGS']['max_failed_gens'])
stories_per_candidate_goal = int(config['STORY GENERATION - GEN SETTINGS']['stories_per_candidate_goal'])
story_words = int(config['STORY GENERATION - GEN SETTINGS']['story_words'])
bias_phrases = ast.literal_eval(config['STORY GENERATION - GEN SETTINGS']['bias_phrases'])
model_class, model_attr = config['STORY GENERATION - GEN SETTINGS']['model'].split('.')
model = getattr(globals()[model_class], model_attr)
max_context = int(config['STORY GENERATION - GEN SETTINGS']['max_context'])
max_gen_length = int(config['STORY GENERATION - GEN SETTINGS']['max_gen_length'])
prompt_prefix = parse_config_value(config['STORY GENERATION - GEN SETTINGS']['prompt_prefix'])
prompt_suffix = parse_config_value(config['STORY GENERATION - GEN SETTINGS']['prompt_suffix'])

# Access the Preset Configuration
preset_method = config['STORY GENERATION - PRESET']['preset_method']
preset_name = config['STORY GENERATION - PRESET']['preset_name']

if preset_method == "custom":
    preset_stop_sequences = ast.literal_eval(config['STORY GENERATION - PRESET']['preset_stop_sequences'])
    preset_temperature = float(config['STORY GENERATION - PRESET']['preset_temperature'])
    preset_max_length = int(config['STORY GENERATION - PRESET']['preset_max_length'])
    preset_min_length = int(config['STORY GENERATION - PRESET']['preset_min_length'])
    preset_top_k = int(config['STORY GENERATION - PRESET']['preset_top_k'])
    preset_top_a = float(config['STORY GENERATION - PRESET']['preset_top_a'])
    preset_top_p = float(config['STORY GENERATION - PRESET']['preset_top_p'])
    preset_typical_p = float(config['STORY GENERATION - PRESET']['preset_typical_p'])
    preset_tail_free_sampling = float(config['STORY GENERATION - PRESET']['preset_tail_free_sampling'])
    preset_repetition_penalty = float(config['STORY GENERATION - PRESET']['preset_repetition_penalty'])
    preset_repetition_penalty_range = int(config['STORY GENERATION - PRESET']['preset_repetition_penalty_range'])
    preset_repetition_penalty_slope = float(config['STORY GENERATION - PRESET']['preset_repetition_penalty_slope'])
    preset_repetition_penalty_frequency = float(config['STORY GENERATION - PRESET']['preset_repetition_penalty_frequency'])
    preset_repetition_penalty_presence = float(config['STORY GENERATION - PRESET']['preset_repetition_penalty_presence'])
    preset_repetition_penalty_whitelist = ast.literal_eval(config['STORY GENERATION - PRESET']['preset_repetition_penalty_whitelist'])
    preset_repetition_penalty_default_whitelist = config['STORY GENERATION - PRESET']['preset_repetition_penalty_default_whitelist'] == 'True'
    preset_length_penalty = float(config['STORY GENERATION - PRESET']['preset_length_penalty'])
    preset_diversity_penalty = float(config['STORY GENERATION - PRESET']['preset_diversity_penalty'])
    preset_order = ast.literal_eval(config['STORY GENERATION - PRESET']['preset_order'])
    preset_phrase_rep_pen = config['STORY GENERATION - PRESET']['preset_phrase_rep_pen']

    preset = Preset(name=preset_name, model=model, settings={
        'temperature': preset_temperature,
        'max_length': preset_max_length,
        'min_length': preset_min_length,
        'top_k': preset_top_k,
        'top_a': preset_top_a,
        'top_p': preset_top_p,
        'typical_p': preset_typical_p,
        'tail_free_sampling': preset_tail_free_sampling,
        'repetition_penalty': preset_repetition_penalty,
        'repetition_penalty_range': preset_repetition_penalty_range,
        'repetition_penalty_slope': preset_repetition_penalty_slope,
        'repetition_penalty_frequency': preset_repetition_penalty_frequency,
        'repetition_penalty_presence': preset_repetition_penalty_presence,
        'repetition_penalty_whitelist': preset_repetition_penalty_whitelist,
        'repetition_penalty_default_whitelist': preset_repetition_penalty_default_whitelist,
        'length_penalty': preset_length_penalty,
        'diversity_penalty': preset_diversity_penalty,
        'order': preset_order,
        'phrase_rep_pen': preset_phrase_rep_pen,
    })

elif preset_method == "official":
    preset = preset_name  # We'll use this string to get the official preset in gen_story
else:
    raise ValueError(f"Invalid preset_method: {preset_method}. Must be 'custom' or 'official'.")

auth = False
env = os.environ

# Init variable for login method
if auth_method == "enter_key":
    auth = input("Enter your NovelAI access key: ")
if auth_method == "enter_token":
    auth = input("Enter your NovelAI access token: ")
elif auth_method == "enter_login":
    auth = {}
    auth["user"] = input("Enter your NovelAI username: ")
    auth["pw"] = input("Enter your NovelAI password: ")
elif auth_method == "env_key":
    auth = env["NAI_KEY"]
elif auth_method == "env_token":
    auth = env["NAI_TOKEN"]
elif auth_method == "env_login":
    auth = {}
    auth["user"] = env["NAI_USERNAME"]
    auth["pw"] = env["NAI_PASSWORD"]
else:
    raise RuntimeError(
        "Invalid value for 'auth_method'. Must be one of 'enter_key', 'enter_token', 'enter_login', 'env_key', 'env_token' or 'env_login"
    )

# Define necessary functions
async def nai_login(api, auth_method, auth):
    """
    Log in to the NovelAI API using the specified authentication method and credentials.
    """
    if auth_method == "enter_key" or auth_method == "env_key":
        await api.high_level.login_from_key(auth)
    elif auth_method == "enter_token" or auth_method == "env_token":
        await api.high_level.login_with_token(auth)
    elif auth_method == "enter_login" or auth_method == "env_login":
        await api.high_level.login(auth["user"], auth["pw"])

async def gen_story(api, prompt, model, preset, max_length=max_gen_length):
    """
    Generate a story using the NovelAI API with the specified parameters.
    """
    global_settings = GlobalSettings()
    if isinstance(preset, str):
        preset = Preset.from_official(model, preset)
    preset['max_length'] = min(max_length, max_gen_length)
    gen = await api.high_level.generate(
        prompt, model, preset, global_settings, None, None, None
    )
    generated_text = Tokenizer.decode(model, b64_to_tokens(gen["output"]))
    return generated_text

async def generate_full_story(api, prompt, model, preset, story_words, max_retries=3, verbose=False, logger=None):
    """
    Generate a full story by iteratively generating text and managing the context.
    """
    full_story = prompt
    current_prompt = prompt
    buffer_tokens = 500  # Buffer to prevent constantly hitting max_context

    if verbose:
        logger.info("*" * 50)
        logger.info(f"Initial max_context: {max_context}")
        logger.info(f"Initial prompt length: {len(prompt.split())} words")

    while len(full_story.split()) < story_words:
        remaining_words = story_words - len(full_story.split())
        prompt_tokens = Tokenizer.encode(model, current_prompt)

        if verbose:
            logger.info("*" * 50)
            logger.info(f"Current prompt tokens: {len(prompt_tokens)}")
            logger.info(f"Remaining words: {remaining_words}")

        available_tokens = max_context - len(prompt_tokens) - buffer_tokens
        max_length = min(max(available_tokens, int(remaining_words * 1.33)), max_gen_length)

        if verbose:
            logger.info(f"Available tokens: {available_tokens}")
            logger.info(f"Max length for generation: {max_length}")

        if len(prompt_tokens) + max_length > max_context - buffer_tokens:
            max_length = max_context - len(prompt_tokens) - buffer_tokens - 1
            if verbose:
                logger.info(f"Adjusted max_length to {max_length} to prevent exceeding max_context")

        if max_length < 1:
            logger.info("Cannot generate more tokens without exceeding max_context. Trimming context.")
            # Trim the current prompt to make room for new generation
            trim_tokens = int(max_context * 0.1)  # Trim 10% of max_context instead of 20%
            current_prompt_tokens = prompt_tokens[trim_tokens:]
            current_prompt = Tokenizer.decode(model, current_prompt_tokens)
            continue

        for _ in range(max_retries):
            try:
                generated_text = await gen_story(api, current_prompt, model, preset, max_length)
                full_story += generated_text
                if verbose:
                    logger.info(f"Generated text length: {len(generated_text.split())} words")
                    logger.info(f"Full story length: {len(full_story.split())} words")

                # Update the prompt for the next iteration
                full_story_tokens = Tokenizer.encode(model, full_story)
                if len(full_story_tokens) > max_context - buffer_tokens:
                    # If the full story exceeds max_context - buffer, trim it
                    keep_tokens = int(max_context * 0.9)  # Keep 90% of max_context
                    trimmed_tokens = full_story_tokens[-keep_tokens:]
                    current_prompt = Tokenizer.decode(model, trimmed_tokens)
                    if verbose:
                        logger.info("*" * 50)
                        logger.info("Trimming context for next generation")
                        logger.info(f"Trimmed prompt to {len(trimmed_tokens)} tokens")
                else:
                    current_prompt = full_story
                    if verbose:
                        logger.info("*" * 50)
                        logger.info("Using full story as prompt for next generation.")
                break
            except Exception as e:
                logger.error(f"Error during generation: {e}. Retrying...")
        else:
            logger.error(f"Failed to generate after {max_retries} attempts. Stopping generation.")
            break

    return full_story

def save_story_to_csv(candidate, story, filename):
    """
    Save a generated story to a CSV file.
    """
    df = pd.DataFrame({'candidate': [candidate], 'story': [story]})
    if not filename.exists():
        df.to_csv(filename, index=False, mode='w')
    else:
        df.to_csv(filename, index=False, mode='a', header=False)
    print(f"Saved story for candidate: {candidate} to {filename}")

def setup_logging(run_name):
    """
    Set up logging for the story generation process.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = PROJECT_ROOT / 'logs' / f"{run_name}_story_generation_{timestamp}.log"
    log_filename.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(log_filename))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# Read results
candidates_file = PROJECT_ROOT / 'data' / f'{run_name}_candidates.csv'
if candidates_file.exists():
    df_candidates = pd.read_csv(candidates_file)
    print(f"Loaded {len(df_candidates)} candidates from {candidates_file}")
else:
    print(f"Error: Candidates file {candidates_file} not found.")
    df_candidates = pd.DataFrame(columns=['phrase'])

# Build list of tuples (candidate, prompt)
prompts = [
    (candidate, f"{prompt_prefix}{candidate}{prompt_suffix}")
    for candidate in df_candidates['phrase']
]

async def main():
    """
    Main function to generate stories for candidate phrases using the NovelAI API.
    """
    logger = setup_logging(run_name)
    api = NovelAIAPI()
    await nai_login(api, auth_method, auth)
    total_generations = 0
    unsuccessful_attempts = 0
    stories_filename = PROJECT_ROOT / 'data' / f'{run_name}_stories.csv'

    for candidate, prompt in prompts:
        for story_num in range(stories_per_candidate_goal):
            while True:
                try:
                    total_generations += 1
                    print(f"Generating story {story_num + 1}/{stories_per_candidate_goal} for candidate: {candidate}")
                    print("Current Full Story:")
                    logger.info(f"Generating story {story_num + 1}/{stories_per_candidate_goal} for candidate: {candidate}")
                    
                    story = await asyncio.wait_for(
                        generate_full_story(api, prompt, model, preset, story_words, verbose=True, logger=logger),
                        timeout=generation_timeout
                    )
                    
                    save_story_to_csv(candidate, story, stories_filename)
                    logger.info(f"Saved story for candidate: {candidate}")
                    unsuccessful_attempts = 0
                    break
                except asyncio.TimeoutError:
                    print("Generation took too long. Retrying...")
                    logger.warning("Generation took too long. Retrying...")
                    unsuccessful_attempts += 1
                    if unsuccessful_attempts >= max_failed_gens:
                        print(f"{max_failed_gens} unsuccessful generation attempts. Skipping this story.")
                        logger.error(f"{max_failed_gens} unsuccessful generation attempts. Skipping this story.")
                        break
                except Exception as e:
                    print(f"Error: {e}")
                    logger.error(f"Error: {e}")
                    unsuccessful_attempts += 1
                    if unsuccessful_attempts >= max_failed_gens:
                        print(f"{max_failed_gens} unsuccessful generation attempts. Skipping this story.")
                        logger.error(f"{max_failed_gens} unsuccessful generation attempts. Skipping this story.")
                        break
            time.sleep(delay_time)
    print("Story generation complete!")
    logger.info("Story generation complete!")

# Run the main function
asyncio.run(main())
