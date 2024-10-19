import asyncio
from novelai_api.NovelAI_API import NovelAIAPI
from novelai_api.GlobalSettings import GlobalSettings
from novelai_api.Preset import Model, Preset
from novelai_api.utils import b64_to_tokens
from novelai_api.Tokenizer import Tokenizer

async def nai_login(api, username, password):
    try:
        await api.high_level.login(username, password)
    except Exception as e:
        raise Exception(f"Login failed: {str(e)}")

async def generate_text(api, model, prompt):
    # Select the appropriate preset based on the model
    preset_name = "Carefree" if model == Model.Kayra else "Golden Arrow"
    
    try:
        preset = Preset.from_official(model, preset_name)
    except Exception as e:
        print(f"Error creating preset for {model}: {e}")
        preset = Preset(model)  # Fallback to default preset
    
    global_settings = GlobalSettings()
    
    try:
        gen = await api.high_level.generate(
            prompt, model, preset, global_settings
        )
        
        # Decode the output
        tokens = b64_to_tokens(gen["output"])
        text = Tokenizer.decode(model, tokens)
        return text
    except Exception as e:
        raise Exception(f"Error generating text: {str(e)}")

async def main():
    api = NovelAIAPI()

    username = input("Enter your NovelAI username: ")
    password = input("Enter your NovelAI password: ")

    try:
        await nai_login(api, username, password)
        print("Login successful!")
    except Exception as e:
        print(f"Error: {e}")
        return

    prompt = "Once upon a time, in a land far away,"

    for model in [Model.Erato, Model.Kayra]:
        print(f"\nGenerating with {model.name} model:")
        try:
            output = await generate_text(api, model, prompt)
            print(f"Prompt: {prompt}")
            print(f"{model.name} output: {output}\n")
        except Exception as e:
            print(f"Error generating with {model.name}: {e}")
            print(f"Model details: {model}")
            print(f"API status: {api.high_level._api._authenticated}")

if __name__ == "__main__":
    asyncio.run(main())
