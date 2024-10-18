# Standard library imports
import asyncio
import re
import time
import os
import configparser
import ast
from datetime import datetime
import logging
from pathlib import Path

# Third-party imports
import pandas as pd
from novelai_api.NovelAI_API import NovelAIAPI
from novelai_api.Preset import Model, Preset
from novelai_api.GlobalSettings import GlobalSettings
from novelai_api.Tokenizer import Tokenizer
from novelai_api.utils import b64_to_tokens
from novelai_api.BiasGroup import BiasGroup

# Import shared functions from common_utils.py
from common_utils import parse_config_value, nai_login, setup_logging

# Define constants
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / 'config'

# Read Settings
config_file = CONFIG_DIR / 'config_character_types_kayra_test.ini'
config = configparser.ConfigParser()
config.read(config_file)

# Access the General Settings
run_name = config['GENERAL']['run_name']
auth_method = config['GENERAL']['auth_method']

# Access the Generation Settings
delay_time = int(config['CANDIDATE GENERATION - GEN SETTINGS']['delay_time'])
generation_timeout = int(config['CANDIDATE GENERATION - GEN SETTINGS']['generation_timeout'])
max_failed_gens = int(config['CANDIDATE GENERATION - GEN SETTINGS']['max_failed_gens'])
candidates_goal = int(config['CANDIDATE GENERATION - GEN SETTINGS']['candidates_goal'])
bias_strength_inc = float(config['CANDIDATE GENERATION - GEN SETTINGS']['bias_strength_inc'])
bias_phrases = ast.literal_eval(config['CANDIDATE GENERATION - GEN SETTINGS']['bias_phrases'])
model_class, model_attr = config['CANDIDATE GENERATION - GEN SETTINGS']['model'].split('.')
model = getattr(globals()[model_class], model_attr)

# Handle 'prompts' as either a single string or a list of strings
prompts_raw = parse_config_value(config['CANDIDATE GENERATION - GEN SETTINGS']['prompts'])
prompts = [prompts_raw] if isinstance(prompts_raw, str) else prompts_raw

stop_sequences = ast.literal_eval(config['CANDIDATE GENERATION - GEN SETTINGS']['stop_sequences'])
checkpoint_interval = int(config['CANDIDATE GENERATION - GEN SETTINGS']['checkpoint_interval'])

# Access the Preset Configuration
preset_method = config['CANDIDATE GENERATION - PRESET']['preset_method']
preset_name = config['CANDIDATE GENERATION - PRESET']['preset_name']

if preset_method == "custom":
    preset_stop_sequences = ast.literal_eval(config['CANDIDATE GENERATION - PRESET']['preset_stop_sequences'])
    preset_temperature = float(config['CANDIDATE GENERATION - PRESET']['preset_temperature'])
    preset_max_length = int(config['CANDIDATE GENERATION - PRESET']['preset_max_length'])
    preset_min_length = int(config['CANDIDATE GENERATION - PRESET']['preset_min_length'])
    preset_top_k = int(config['CANDIDATE GENERATION - PRESET']['preset_top_k'])
    preset_top_a = float(config['CANDIDATE GENERATION - PRESET']['preset_top_a'])
    preset_top_p = float(config['CANDIDATE GENERATION - PRESET']['preset_top_p'])
    preset_typical_p = float(config['CANDIDATE GENERATION - PRESET']['preset_typical_p'])
    preset_tail_free_sampling = float(config['CANDIDATE GENERATION - PRESET']['preset_tail_free_sampling'])
    preset_repetition_penalty = float(config['CANDIDATE GENERATION - PRESET']['preset_repetition_penalty'])
    preset_repetition_penalty_range = int(config['CANDIDATE GENERATION - PRESET']['preset_repetition_penalty_range'])
    preset_repetition_penalty_slope = float(config['CANDIDATE GENERATION - PRESET']['preset_repetition_penalty_slope'])
    preset_repetition_penalty_frequency = float(config['CANDIDATE GENERATION - PRESET']['preset_repetition_penalty_frequency'])
    preset_repetition_penalty_presence = float(config['CANDIDATE GENERATION - PRESET']['preset_repetition_penalty_presence'])
    preset_repetition_penalty_whitelist = ast.literal_eval(config['CANDIDATE GENERATION - PRESET']['preset_repetition_penalty_whitelist'])
    preset_repetition_penalty_default_whitelist = config['CANDIDATE GENERATION - PRESET']['preset_repetition_penalty_default_whitelist'] == 'True'
    preset_length_penalty = float(config['CANDIDATE GENERATION - PRESET']['preset_length_penalty'])
    preset_diversity_penalty = float(config['CANDIDATE GENERATION - PRESET']['preset_diversity_penalty'])
    preset_order = ast.literal_eval(config['CANDIDATE GENERATION - PRESET']['preset_order'])
    preset_phrase_rep_pen = config['CANDIDATE GENERATION - PRESET']['preset_phrase_rep_pen']

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
    preset = preset_name  # We'll use this string to get the official preset in gen_attg_candidate
else:
    raise ValueError(f"Invalid preset_method: {preset_method}. Must be 'custom' or 'official'.")

auth = False
env = os.environ

# Init variable for login method
if auth_method == "enter_key":
    auth = input("Enter your NovelAI access key: ")
elif auth_method == "enter_token":
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
        "Invalid value for 'auth_method'. Must be one of 'enter_key', 'enter_token', 'enter_login', 'env_key', 'env_token' or 'env_login'"
    )

async def gen_attg_candidate(
    model=Model.Clio,
    preset="Edgewise",
    prompt="[ Genre:",
    stop_sequences=[",", ";", " ]","\n"],
    cut_stop_seq=True,
    auth_method="env_token",
    auth=None,
    bias_groups=None,
):
    """
    Generate a candidate phrase using the NovelAI API with the specified parameters.
    """
    # Initialize the NovelAI API
    api = NovelAIAPI()

    try:
        # Ensure you're logged in
        await nai_login(api, auth_method, auth)

        # If preset is a string, get the official preset with that name for the specified model
        if isinstance(preset, str):
            preset = Preset.from_official(model, preset)

        # Tokenize the stop sequences and set them for the preset
        stop_sequences_tokenized = [
            Tokenizer.encode(model, seq) for seq in stop_sequences
        ]
        preset["stop_sequences"] = stop_sequences_tokenized

        # Create default global settings
        global_settings = GlobalSettings()

        gen = await api.high_level.generate(
            prompt, model, preset, global_settings, None, bias_groups, None
        )

        # After generating the text, keep the stop sequence
        generated_text = Tokenizer.decode(model, b64_to_tokens(gen["output"]))
        if not cut_stop_seq:
            # Find the first occurrence of any stop sequence
            stop_index = len(generated_text)
            for seq in stop_sequences:
                seq_index = generated_text.find(seq)
                if seq_index != -1 and seq_index < stop_index:
                    stop_index = seq_index

            # Truncate the generated text at the first stop sequence
            generated_text = generated_text[:stop_index + 1]  # +1 to include the stop character
        else:
            # If cut_stop_seq is True, remove the stop sequences as before
            for seq in stop_sequences:
                generated_text = re.sub(
                    re.escape(seq) + "$", "", generated_text
                ).strip()

        return generated_text

    except Exception as e:
        raise Exception(f"Error generating text: {e}")

def update_bias_groups(phrase, bias_phrase_dict, bias_strength_inc, bias_groups):
    """
    Update the bias groups based on the generated phrase and bias strength increment.
    """
    # Update the bias strength for the phrase or add it if it's not in the dict
    if phrase in bias_phrase_dict:
        bias_phrase_dict[phrase] += bias_strength_inc
    else:
        bias_phrase_dict[phrase] = bias_strength_inc

    # Clear the existing bias groups
    bias_groups.clear()

    # Regenerate the bias groups based on the updated bias_phrase_dict
    for phrase, strength in bias_phrase_dict.items():
        bg = BiasGroup(strength)
        bg.add(phrase)
        bias_groups.append(bg)

def load_existing_candidates(run_name):
    """
    Load existing candidate phrases and their bias strengths from a CSV file.
    """
    filename = PROJECT_ROOT / 'data' / f"{run_name}_candidates.csv"
    if filename.exists():
        df = pd.read_csv(filename)
        bias_phrases = dict(zip(df['phrase'], df['last_bias']))
        return df, bias_phrases
    else:
        return pd.DataFrame(columns=["phrase", "count", "last_bias"]), {}

def update__candidates_run_info(run_name, settings, terms_generated, terms_added, status, start_time, is_checkpoint=False):
    """
    Update the candidate generation run information and save it to a CSV file.
    """
    filename = PROJECT_ROOT / 'data' / f"{run_name}__candidates_run_info.csv"
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    _candidates_run_info = pd.DataFrame({
        'timestamp': [end_time],
        'terms_generated': [terms_generated],
        'terms_added': [terms_added],
        'status': [status],
        'duration': [duration],
        'is_checkpoint': [is_checkpoint],
        **settings
    })
    if filename.exists():
        existing_info = pd.read_csv(filename)
        # Remove previous checkpoints if this is a new checkpoint
        if is_checkpoint:
            existing_info = existing_info[existing_info['is_checkpoint'] == False]
        updated_info = pd.concat([existing_info, _candidates_run_info], ignore_index=True)
    else:
        updated_info = _candidates_run_info
    updated_info.to_csv(filename, index=False)

async def main():
    """
    Main function to generate candidate phrases using the NovelAI API.
    """
    # Setup logging
    logger = setup_logging(run_name, 'candidates', PROJECT_ROOT)

    # Load existing results if available
    df, bias_phrases = load_existing_candidates(run_name)

    # Initialize bias groups
    bias_groups = []
    for phrase, strength in bias_phrases.items():
        bg = BiasGroup(strength)
        bg.add(phrase)
        bias_groups.append(bg)

    # Counter for total generations and unsuccessful attempts
    total_generations = 0
    unsuccessful_attempts = 0
    terms_added = 0

    # Initialize settings_data outside the loop
    settings_data = {
        "auth_method": auth_method,
        "candidates_goal": candidates_goal,
        "bias_strength_inc": bias_strength_inc,
        "model": str(model),
        "preset_name": preset_name,
        "preset_method": preset_method,
        "preset_settings": str(preset._settings) if preset_method == "custom" else "N/A",
        "bias_phrases": str(bias_phrases),
        "checkpoint_interval": checkpoint_interval
    }

    start_time = datetime.now()

    # Loop until you have candidate_goal unique phrases
    try:
        while len(df) < candidates_goal:
            for prompt_index, prompt in enumerate(prompts, 1):
                total_generations += 1

                prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
                print(f"Gen {total_generations}: Trying to gen phrase {len(df)+1}/{candidates_goal}...")
                print(f"Using prompt {prompt_index}/{len(prompts)}: {prompt_preview}")
                logger.info(f"Gen {total_generations}: Trying to gen phrase {len(df)+1}/{candidates_goal}...")
                logger.info(f"Using prompt {prompt_index}/{len(prompts)}: {prompt_preview}")

                try:
                    phrase = await asyncio.wait_for(
                        gen_attg_candidate(
                            model=model,
                            preset=preset,
                            prompt=prompt,
                            stop_sequences=stop_sequences,
                            cut_stop_seq=False,
                            auth_method=auth_method,
                            auth=auth,
                            bias_groups=bias_groups,
                        ),
                        timeout=generation_timeout,
                    )

                    # Check if the phrase is already in the DataFrame
                    if phrase in df["phrase"].values:
                        df.loc[df["phrase"] == phrase, "count"] += 1
                        df.loc[df["phrase"] == phrase, "last_bias"] = bias_phrases.get(phrase, 0)
                        print(f"Phrase '{phrase}' already exists. Incrementing count and changing bias by {bias_strength_inc}.")
                        logger.info(f"Phrase '{phrase}' already exists. Incrementing count and changing bias by {bias_strength_inc}.")

                        # Update the bias groups since the phrase was generated again
                        update_bias_groups(phrase, bias_phrases, bias_strength_inc, bias_groups)
                    else:
                        new_row = pd.DataFrame({"phrase": [phrase], "count": [1], "last_bias": [bias_phrases.get(phrase, 0)]})
                        df = pd.concat([df, new_row], ignore_index=True)
                        print(f"Added new phrase: '{phrase}'")
                        logger.info(f"Added new phrase: '{phrase}'")
                        terms_added += 1

                    # Reset the unsuccessful_attempts counter if generation was successful
                    unsuccessful_attempts = 0

                    # Store results and settings
                    filename_candidates = PROJECT_ROOT / 'data' / f"{run_name}_candidates.csv"
                    df.to_csv(filename_candidates, index=False)

                    # Update settings_data with the latest bias_phrases
                    settings_data["bias_phrases"] = str(bias_phrases)

                    print(f"Saved progress to {filename_candidates}.")
                    logger.info(f"Saved progress to {filename_candidates}.")

                    # Save checkpoint every checkpoint_interval generations
                    if total_generations % checkpoint_interval == 0:
                        update__candidates_run_info(run_name, settings_data, total_generations, terms_added, "ongoing", start_time, is_checkpoint=True)
                        print(f"Checkpoint saved at {total_generations} generations.")
                        logger.info(f"Checkpoint saved at {total_generations} generations.")

                except asyncio.TimeoutError:
                    print("Generation took too long. Retrying...")
                    logger.warning("Generation took too long. Retrying...")
                    unsuccessful_attempts += 1
                    if unsuccessful_attempts >= max_failed_gens:
                        print(f"{max_failed_gens} unsuccessful generation attempts. Aborting candidate search.")
                        logger.error(f"{max_failed_gens} unsuccessful generation attempts. Aborting candidate search.")
                        update__candidates_run_info(run_name, settings_data, total_generations, terms_added, "aborted", start_time)
                        return
                except Exception as e:
                    if "Anonymous quota reached" in str(e):
                        print(f"Error: {e}")
                        print("Anonymous rate limit reached. This indicates you are not properly authenticated. Check your authentication method. Aborting candidate search.")
                        logger.error(f"Error: {e}")
                        logger.error("Anonymous rate limit reached. This indicates you are not properly authenticated. Check your authentication method. Aborting candidate search.")
                        update__candidates_run_info(run_name, settings_data, total_generations, terms_added, "aborted", start_time)
                        return
                    else:
                        print(f"Error: {e}")
                        print("Aborting candidate search")
                        logger.error(f"Error: {e}")
                        logger.error("Aborting candidate search")
                        update__candidates_run_info(run_name, settings_data, total_generations, terms_added, "aborted", start_time)
                        return

                # Wait for delay_time seconds before the next generation attempt
                time.sleep(delay_time)

                if len(df) >= candidates_goal:
                    break

        # Final update of run info if completed successfully
        if len(df) >= candidates_goal:
            update__candidates_run_info(run_name, settings_data, total_generations, terms_added, "completed", start_time)

        print("\nCandidate search complete!")
        print("Top 10 terms:")
        print(df.sort_values(by="count", ascending=False).head(10))
        logger.info("\nCandidate search complete!")
        logger.info("Top 10 terms:")
        logger.info(df.sort_values(by="count", ascending=False).head(10))

    except KeyboardInterrupt:
        print("\nRun interrupted by user.")
        logger.info("Run interrupted by user.")
        update__candidates_run_info(run_name, settings_data, total_generations, terms_added, "interrupted", start_time)

# Run the main function
asyncio.run(main())
