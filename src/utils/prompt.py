DIRECT_PROMPT = '''Question: {question}
Choices:
{choices}
Output final answer (option letter) directly.
'''

WEATHER_R1_PROMPT = '''Question: {question}
Choices:
{choices}
Answer with the option's letter from the given choices.
You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags.
'''

LOGIC_REWARD_PROMPT = """Your task is to select the option best supported by the given reasoning process.
Directly output the uppercase letter of the selected option. If the reasoning process does not correspond to any of the options, output "Cannot be determined".
[Input]: {question_and_options}\nReasoning process: {reasoning_process}
[Output]: """